# emulator.py

"""
Xbox 360 Emulator Main Class
This version can dispatch to the correct parser (XEX or PE) based on file type.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any

from .cpu import XenonCPU
from .memory import MemoryManager
from .kernel import KernelInterface
from .gpu import XenosGPU
from .filesystem import FileSystem
from .debugger import EmulatorDebugger
from .xex import XEXParser
from .pe import PEParser # Import the new PE parser

logger = logging.getLogger(__name__)

class Xbox360Emulator:
    def __init__(self, config: Dict = None):
        self.config = config if config else {}
        self.running = False
        self.paused = False
        
        logger.info("Initializing Xbox 360 Emulator...")
        
        self.memory = MemoryManager()
        self.gpu = XenosGPU(self.memory)
        self.cpu = XenonCPU(self.memory)
        self.filesystem = FileSystem(Path(self.config.get('paths', {}).get('game_directory', 'emu_files/')))
        self.kernel = KernelInterface(self.config.get('system', {}).get('kernel_path'), self.memory, self)
        self.debugger = EmulatorDebugger(self)
        
        self.emulation_thread: Optional[threading.Thread] = None
        self._register_device_mmio()
        logger.info("Xbox 360 Emulator initialized successfully")

    def _register_device_mmio(self):
        """Registers MMIO handlers for all hardware devices."""
        gpu_handlers = self.gpu.get_mmio_handlers()
        for reg_addr, handlers in gpu_handlers.items():
            self.memory.register_mmio_handler(range(reg_addr, reg_addr + 4), handlers[0], handlers[1])
        logger.info("MMIO handlers for all devices registered.")

    def reset(self):
        """Reset emulator to initial state"""
        logger.info("Resetting emulator...")
        if self.running:
            self.stop()
        self.cpu.reset()
        self.gpu.reset()
        logger.info("Emulator reset complete")

    def load_game(self, game_path: str) -> bool:
        """
        Loads a game executable, automatically detecting if it is a XEX or PE file.
        """
        logger.info(f"Attempting to load game: {game_path}")
        try:
            with open(game_path, 'rb') as f:
                game_data = f.read()
        except FileNotFoundError:
            logger.error(f"Game file not found: {game_path}")
            return False

        # --- Parser Dispatch Logic ---
        parser = None
        if len(game_data) > 4 and game_data[:4] == b'XEX2':
            logger.info("XEX2 file format detected.")
            parser = XEXParser(game_data)
        elif len(game_data) > 2 and game_data[:2] == b'MZ':
            logger.info("PE/EXE file format detected.")
            parser = PEParser(game_data)
        else:
            logger.error("Unknown or unsupported executable format. Must be XEX or PE.")
            return False

        if not parser or not parser.is_valid:
            logger.error(f"Failed to parse executable file: {parser.error_message if parser else 'N/A'}")
            return False
            
        exec_info = parser.get_execution_info()
        base_file = parser.get_base_file()

        if not exec_info or not base_file:
            logger.error("Could not retrieve execution info or base file from parser.")
            return False

        logger.info(f"Loading base file into memory at 0x{exec_info.base_address:08X}")
        self.memory.write_memory(exec_info.base_address, base_file)

        for core in self.cpu.cores:
            for thread_regs in core.threads:
                thread_regs.pc = exec_info.entry_point
                thread_regs.gpr[1] = 0x81000000  # Initial stack pointer

        logger.info(f"Game loaded. Entry Point: 0x{exec_info.entry_point:08X}")
        return True

    def start(self):
        """Starts the main emulation thread."""
        if self.running: return
        logger.info("Starting emulation...")
        self.running = True
        self.emulation_thread = threading.Thread(target=self._emulation_loop, daemon=True)
        self.emulation_thread.start()

    def stop(self):
        """Stops the emulation thread."""
        if not self.running: return
        logger.info("Stopping emulation...")
        self.running = False
        self.cpu.stop()
        if self.emulation_thread and self.emulation_thread.is_alive():
            self.emulation_thread.join(timeout=2.0)
        logger.info("Emulation stopped.")

    def _emulation_loop(self):
        """Main emulation loop."""
        logger.info("Emulation loop started.")
        while self.running:
            self.cpu.run(100000) # Run in slices
            self.gpu.present()
            time.sleep(0.016) # ~60 FPS
        logger.info("Emulation loop ended.")
