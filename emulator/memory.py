# memory.py

"""
Xbox 360 Memory Management
Handles memory mapping, allocation, and protection. This version adds support
for Memory-Mapped I/O (MMIO) to allow hardware components like the GPU to
be controlled by memory writes from the CPU.
"""

import struct
import logging
from typing import Optional, Dict, List, Tuple, Callable

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, total_memory: int = 512 * 1024 * 1024):
        self.total_memory = total_memory
        # Main system RAM
        self.ram = bytearray(total_memory)
        
        # MMIO handlers: map address range to a device's read/write methods
        self.mmio_handlers: List[Tuple[range, Tuple[Optional[Callable], Optional[Callable]]]] = []
        
        logger.info(f"Initialized Memory Manager with {total_memory // (1024*1024)}MB RAM and MMIO support.")
        
    def register_mmio_handler(self, address_range: range, read_handler: Optional[Callable], write_handler: Optional[Callable]):
        """Registers a device's read/write methods for a specific address range."""
        self.mmio_handlers.append((address_range, (read_handler, write_handler)))
        logger.info(f"Registered MMIO handler for range 0x{address_range.start:08X} - 0x{address_range.stop-1:08X}")

    def _find_mmio_handler(self, address: int):
        """Finds a registered MMIO handler for a given address."""
        for addr_range, handlers in self.mmio_handlers:
            if address in addr_range:
                return handlers
        return None, None

    def translate_address(self, virtual_addr: int) -> int:
        """Translate virtual address to physical address."""
        # This is a simplified translation. A real system would have a complex page table.
        if 0x80000000 <= virtual_addr < 0xA0000000: # Kernel space
            return virtual_addr - 0x80000000
        # Other ranges would be handled here
        return virtual_addr # Assume physical for now for other ranges

    def read_memory(self, address: int, size: int) -> bytes:
        """Read memory from an address, dispatching to MMIO if necessary."""
        read_handler, _ = self._find_mmio_handler(address)
        if read_handler:
            # For simplicity, MMIO reads are 4 bytes. A real system would handle size.
            value = read_handler(address)
            return struct.pack('>I', value)
            
        try:
            phys_addr = self.translate_address(address)
            if phys_addr + size > self.total_memory:
                raise IndexError("Memory read out of bounds")
            return self.ram[phys_addr : phys_addr + size]
        except IndexError as e:
            logger.warning(f"Memory read error at 0x{address:08X}: {e}")
            return b'\x00' * size
            
    def write_memory(self, address: int, data: bytes):
        """Write memory to an address, dispatching to MMIO if necessary."""
        _, write_handler = self._find_mmio_handler(address)
        if write_handler:
            # For simplicity, MMIO writes are 4 bytes.
            if len(data) == 4:
                value = struct.unpack('>I', data)[0]
                write_handler(address, value)
            return

        try:
            phys_addr = self.translate_address(address)
            if phys_addr + len(data) > self.total_memory:
                raise IndexError("Memory write out of bounds")
            self.ram[phys_addr : phys_addr + len(data)] = data
        except IndexError as e:
            logger.warning(f"Memory write error at 0x{address:08X}: {e}")

    # --- Convenience methods for reading/writing specific data types ---
    def read_uint32(self, address: int) -> int:
        data = self.read_memory(address, 4)
        return struct.unpack('>I', data)[0]
        
    def write_uint32(self, address: int, value: int):
        self.write_memory(address, struct.pack('>I', value & 0xFFFFFFFF))
        
    def read_uint16(self, address: int) -> int:
        data = self.read_memory(address, 2)
        return struct.unpack('>H', data)[0]
        
    def write_uint16(self, address: int, value: int):
        self.write_memory(address, struct.pack('>H', value & 0xFFFF))
        
    def read_uint8(self, address: int) -> int:
        data = self.read_memory(address, 1)
        return data[0]
        
    def write_uint8(self, address: int, value: int):
        self.write_memory(address, bytes([value & 0xFF]))

    def dump_memory(self, start: int, size: int) -> str:
        """Dump memory contents as a hex string for debugging."""
        data = self.read_memory(start, size)
        lines = []
        for i in range(0, len(data), 16):
            addr = start + i
            hex_bytes = ' '.join(f'{b:02X}' for b in data[i:i+16])
            ascii_chars = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in data[i:i+16])
            lines.append(f'{addr:08X}: {hex_bytes:<48} {ascii_chars}')
        return '\n'.join(lines)
