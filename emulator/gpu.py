# gpu.py

"""
Xbox 360 Xenos GPU Emulation
This version implements a real command buffer processor and a fast, NumPy-based
software rendering pipeline to render game data, not simulations. This corrected
version fixes a bug in the SET_CONSTANT command processing.
"""

import logging
import struct
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from PIL import Image

logger = logging.getLogger(__name__)

# --- GPU Data Structures ---

class VertexFormat:
    """Describes the layout of a single vertex in memory."""
    def __init__(self):
        self.stride = 0
        self.elements = [] # Tuples of (offset, type, usage)

    def add_element(self, type_str: str, usage: str):
        # A real implementation would parse complex format specifiers.
        # We'll use a simplified version.
        size_map = {'f32': 4, 'u8': 1}
        count_map = {'position': 3, 'color0': 4, 'texcoord0': 2}
        
        size = size_map.get(type_str, 4) * count_map.get(usage, 1)
        self.elements.append({'offset': self.stride, 'type': type_str, 'usage': usage, 'size': size})
        self.stride += size

class GPUState:
    """Holds the entire state of the GPU rendering pipeline."""
    def __init__(self):
        self.constants: Dict[int, np.ndarray] = {}
        self.textures: Dict[int, 'Texture'] = {}
        self.vertex_format = VertexFormat()
        self.active_texture_sampler = 0
        
        # Pointers to data in main memory
        self.index_buffer_ptr = 0
        self.vertex_buffer_ptr = 0

        # Default transformation matrices
        self.constants[0] = np.identity(4, dtype=np.float32) # World
        self.constants[4] = np.identity(4, dtype=np.float32) # View
        self.constants[8] = np.identity(4, dtype=np.float32) # Projection

class Texture:
    """Represents a texture loaded into GPU memory."""
    def __init__(self, width, height, format, data):
        self.width = width
        self.height = height
        self.format = format # e.g., 'DXT1', 'A8R8G8B8'
        # Ensure data is correctly shaped for RGBA
        self.data = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))

    def sample(self, u: float, v: float) -> np.ndarray:
        """Fast texture sampling with nearest-neighbor."""
        x = int(u * (self.width - 1) + 0.5) % self.width
        y = int(v * (self.height - 1) + 0.5) % self.height
        return self.data[y, x] / 255.0

class XenosGPU:
    """Xenos GPU Emulator with a real command processor."""
    
    # GPU Register addresses
    REG_CP_RB_BASE = 0xEC800100
    REG_CP_RB_RPTR = 0xEC800108
    REG_CP_RB_WPTR = 0xEC80010C
    
    def __init__(self, memory_manager, width: int = 1280, height: int = 720):
        self.memory = memory_manager
        self.width = width
        self.height = height
        
        # Frame and depth buffers
        self.front_buffer = np.zeros((height, width, 4), dtype=np.uint8)
        self.back_buffer = np.zeros((height, width, 4), dtype=np.uint8)
        self.depth_buffer = np.full((height, width), float('inf'), dtype=np.float32)
        
        # GPU state
        self.registers = {}
        self.state = GPUState()
        self.command_buffer_base = 0
        self.command_buffer_rptr = 0
        
        # Performance stats
        self.frame_count = 0
        self.triangles_rendered = 0
        
        self.setup_default_state()
        logger.info(f"Initialized Xenos GPU with Command Processor")

    def reset(self):
        """Resets the GPU to a default state."""
        self.registers.clear()
        self.command_buffer_base = 0
        self.command_buffer_rptr = 0
        self.frame_count = 0
        self.triangles_rendered = 0
        
        # Clear buffers to default state
        self.back_buffer.fill(0) # Black with 0 alpha
        self.depth_buffer.fill(float('inf'))
        self.present() # Swap the cleared back buffer to the front

        self.setup_default_state() # Re-apply default matrices and formats
        logger.info("GPU reset to initial state.")

    def setup_default_state(self):
        """Sets up a default state for rendering."""
        # Default projection matrix
        fov, aspect, near, far = np.pi / 2, self.width / self.height, 0.1, 1000.0
        f = 1.0 / np.tan(fov / 2)
        # Constants start at c0, c4, c8 for W, V, P
        self.state.constants[8] = np.array([
            [f / aspect, 0, 0, 0], [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ], dtype=np.float32)

        # Default vertex format
        vf = VertexFormat()
        vf.add_element('f32', 'position')
        vf.add_element('u8', 'color0')
        vf.add_element('f32', 'texcoord0')
        self.state.vertex_format = vf

    def get_mmio_handlers(self) -> Dict[int, Tuple[Optional[Callable], Optional[Callable]]]:
        """Returns the MMIO handlers for GPU registers."""
        return {
            self.REG_CP_RB_WPTR: (None, self.write_wptr),
            # Add other register handlers here
        }

    def write_wptr(self, address: int, value: int):
        """MMIO Write Handler for the Command Buffer Write Pointer."""
        self.registers[address] = value
        # The wptr value is an offset in dwords (4 bytes)
        self.process_command_buffer(value)

    def process_command_buffer(self, wptr_dwords: int):
        """Processes command packets from the ring buffer in memory."""
        # Ensure the command buffer base has been set by the game
        if self.command_buffer_base == 0:
            self.command_buffer_base = self.memory.read_uint32(self.REG_CP_RB_BASE)
            if self.command_buffer_base == 0:
                logger.warning("GPU command buffer base not set, cannot process commands.")
                return

        rptr_dwords = self.command_buffer_rptr
        
        while rptr_dwords < wptr_dwords:
            # The address of the command packet in main memory
            cmd_addr = self.command_buffer_base + (rptr_dwords * 4)
            
            try:
                header = self.memory.read_uint32(cmd_addr)
                packet_type = (header >> 30) & 0x3
                count = (header >> 16) & 0x3FFF # Count is in dwords
                opcode = (header >> 8) & 0x7F

                if packet_type == 3: # Type 3 packet
                    self.process_type3_packet(opcode, cmd_addr + 4, count)
                
                rptr_dwords += count + 1 # Advance rptr past packet (header + data)
            except Exception as e:
                logger.error(f"Error processing command buffer at 0x{cmd_addr:08X}: {e}", exc_info=True)
                break
        
        self.command_buffer_rptr = rptr_dwords
        # Update the read pointer register for the game to see
        self.registers[self.REG_CP_RB_RPTR] = rptr_dwords

    def process_type3_packet(self, opcode: int, data_addr: int, count: int):
        """Handles specific Type 3 command opcodes."""
        if opcode == 0x1E: # DRAW_INDX
            if count < 2: return
            num_indices = self.memory.read_uint32(data_addr)
            self.render_indexed_primitive(num_indices)
        elif opcode == 0x22: # SET_CONSTANT
            if count == 0: return
            # The first dword contains the start register and count
            info = self.memory.read_uint32(data_addr)
            start_reg = info & 0x7FF
            num_consts = ((info >> 16) & 0x7FF) + 1
            
            # Each constant is a 4-float vector (16 bytes = 4 dwords)
            # The packet count should match this.
            if count != (num_consts * 4) + 1:
                logger.warning(f"SET_CONSTANT packet size mismatch. Packet count: {count}, expected: {(num_consts * 4) + 1}")
                return

            for i in range(num_consts):
                const_data_addr = data_addr + 4 + (i * 16)
                const_data = self.memory.read_memory(const_data_addr, 16)
                # Constants are organized in banks of 4 vectors. We simplify this.
                self.state.constants[start_reg + i] = np.frombuffer(const_data, dtype='<f4')

    def render_indexed_primitive(self, num_indices: int):
        """Fetches data and renders a primitive."""
        if num_indices == 0: return
        
        # 1. Fetch index data
        index_data = self.memory.read_memory(self.state.index_buffer_ptr, num_indices * 2)
        indices = np.frombuffer(index_data, dtype=np.uint16)

        # 2. Fetch vertex data
        vf = self.state.vertex_format
        if vf.stride == 0:
            logger.error("Vertex format stride is 0, cannot render.")
            return
        max_index = np.max(indices)
        vertex_data_size = (max_index + 1) * vf.stride
        vertex_data = self.memory.read_memory(self.state.vertex_buffer_ptr, vertex_data_size)
        
        # 3. Assemble transformation matrix
        wvp_matrix = self.state.constants[8] @ self.state.constants[4] @ self.state.constants[0]

        # 4. Process triangles
        for i in range(0, num_indices, 3):
            if i + 2 >= len(indices): break
            v_indices = indices[i:i+3]
            
            v0_data = self.unpack_vertex(vertex_data, v_indices[0], vf)
            v1_data = self.unpack_vertex(vertex_data, v_indices[1], vf)
            v2_data = self.unpack_vertex(vertex_data, v_indices[2], vf)

            p0 = self.vertex_shader(v0_data.get('position', np.zeros(3)), wvp_matrix)
            p1 = self.vertex_shader(v1_data.get('position', np.zeros(3)), wvp_matrix)
            p2 = self.vertex_shader(v2_data.get('position', np.zeros(3)), wvp_matrix)
            
            self.rasterize_triangle(p0, p1, p2, v0_data, v1_data, v2_data)
        
        self.triangles_rendered += num_indices // 3

    def unpack_vertex(self, buffer: bytes, index: int, vf: VertexFormat) -> Dict:
        """Extracts a single vertex's attributes from the vertex buffer."""
        base = index * vf.stride
        vertex = {}
        for elem in vf.elements:
            offset = base + elem['offset']
            data = buffer[offset : offset + elem['size']]
            if not data: continue
            
            dtype_map = {'f32': '<f4', 'u8': np.uint8}
            dtype = dtype_map.get(elem['type'], '<f4')
            
            try:
                if elem['usage'] == 'position':
                    vertex['position'] = np.frombuffer(data, dtype=dtype)
                elif elem['usage'] == 'color0':
                    vertex['color'] = np.frombuffer(data, dtype=dtype) / 255.0
                elif elem['usage'] == 'texcoord0':
                    vertex['texcoord'] = np.frombuffer(data, dtype=dtype)
            except ValueError:
                # Handle cases where buffer slice is not a multiple of the item size
                logger.warning(f"Could not unpack vertex element {elem['usage']} due to size mismatch.")
                pass
        return vertex

    def vertex_shader(self, pos: np.ndarray, wvp: np.ndarray) -> np.ndarray:
        """Transforms a vertex from model space to clip space."""
        if pos.shape[0] < 3: return np.array([0,0,0,1])
        pos_h = np.array([pos[0], pos[1], pos[2], 1.0], dtype=np.float32)
        return wvp @ pos_h

    def pixel_shader(self, color: np.ndarray, texcoord: np.ndarray) -> np.ndarray:
        """Determines the final color of a pixel."""
        tex = self.state.textures.get(self.state.active_texture_sampler)
        if tex and texcoord is not None and texcoord.shape[0] >= 2:
            tex_color = tex.sample(texcoord[0], texcoord[1])
            # Ensure color has 4 components (RGBA)
            if color.shape[0] < 4:
                color = np.append(color, [1.0] * (4 - color.shape[0]))
            return color * tex_color
        return color if color is not None else np.array([0,0,0,0])

    def rasterize_triangle(self, p0_clip, p1_clip, p2_clip, v0, v1, v2):
        """Fast NumPy-based triangle rasterizer."""
        points = np.array([p0_clip, p1_clip, p2_clip])
        points_w = points[:, 3]
        
        if np.any(points_w < 0.1): return
        
        points_inv_w = 1.0 / points_w
        points_ndc = points[:, :3] * points_inv_w[:, np.newaxis]
        
        points_screen = np.empty_like(points_ndc)
        points_screen[:, 0] = (points_ndc[:, 0] + 1) * 0.5 * self.width
        points_screen[:, 1] = (1 - points_ndc[:, 1]) * 0.5 * self.height
        points_screen[:, 2] = points_ndc[:, 2]

        min_x, min_y = np.min(points_screen, axis=0)[:2].astype(int)
        max_x, max_y = np.max(points_screen, axis=0)[:2].astype(int)
        min_x, min_y = max(0, min_x), max(0, min_y)
        max_x, max_y = min(self.width, max_x + 1), min(self.height, max_y + 1)
        
        if min_x >= max_x or min_y >= max_y: return

        p = points_screen
        v0_v, v1_v = p[1, :2] - p[0, :2], p[2, :2] - p[0, :2]
        
        x_range = np.arange(min_x, max_x)
        y_range = np.arange(min_y, max_y)
        px, py = np.meshgrid(x_range, y_range)
        
        v2_v = np.stack((px.ravel() - p[0, 0], py.ravel() - p[0, 1]), axis=-1)
        
        dot00 = np.dot(v0_v, v0_v)
        dot01 = np.dot(v0_v, v1_v)
        dot11 = np.dot(v1_v, v1_v)
        
        inv_denom = dot00 * dot11 - dot01 * dot01
        if abs(inv_denom) < 1e-6: return # Degenerate triangle
        inv_denom = 1.0 / inv_denom
        
        dot20 = np.sum(v2_v * v0_v, axis=1)
        dot21 = np.sum(v2_v * v1_v, axis=1)
        
        w1 = (dot11 * dot20 - dot01 * dot21) * inv_denom
        w2 = (dot00 * dot21 - dot01 * dot20) * inv_denom
        w0 = 1.0 - w1 - w2

        mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not np.any(mask): return

        z_interp = w0[mask] * p[0, 2] + w1[mask] * p[1, 2] + w2[mask] * p[2, 2]
        
        y_coords, x_coords = py.ravel()[mask], px.ravel()[mask]
        
        depth_mask = z_interp < self.depth_buffer[y_coords, x_coords]
        if not np.any(depth_mask): return
        
        final_y, final_x = y_coords[depth_mask], x_coords[depth_mask]
        self.depth_buffer[final_y, final_x] = z_interp[depth_mask]
        
        w = np.array([w0[mask][depth_mask], w1[mask][depth_mask], w2[mask][depth_mask]])
        inv_w_interp = w[0] * points_inv_w[0] + w[1] * points_inv_w[1] + w[2] * points_inv_w[2]
        
        # Interpolate attributes
        color_interp = self.interpolate_attribute(w, points_inv_w, inv_w_interp, v0.get('color'), v1.get('color'), v2.get('color'))
        texcoord_interp = self.interpolate_attribute(w, points_inv_w, inv_w_interp, v0.get('texcoord'), v1.get('texcoord'), v2.get('texcoord'))

        for i in range(len(final_x)):
            final_color = self.pixel_shader(color_interp[i] if color_interp is not None else None, texcoord_interp[i] if texcoord_interp is not None else None)
            if final_color is not None:
                self.back_buffer[final_y[i], final_x[i]] = (np.clip(final_color, 0, 1) * 255).astype(np.uint8)

    def interpolate_attribute(self, w, points_inv_w, inv_w_interp, a0, a1, a2):
        """Helper for perspective-correct attribute interpolation."""
        if a0 is None or a1 is None or a2 is None: return None
        return (w[0,:,np.newaxis]*a0*points_inv_w[0] + w[1,:,np.newaxis]*a1*points_inv_w[1] + w[2,:,np.newaxis]*a2*points_inv_w[2]) / inv_w_interp[:,np.newaxis]

    def present(self):
        """Swaps back and front buffers."""
        self.front_buffer, self.back_buffer = self.back_buffer, self.front_buffer
        self.back_buffer.fill(0)
        self.depth_buffer.fill(float('inf'))
        self.frame_count += 1
        
    def get_frame_buffer_image(self) -> Image.Image:
        """Gets the currently visible frame buffer as a PIL Image."""
        return Image.fromarray(self.front_buffer, 'RGBA')
