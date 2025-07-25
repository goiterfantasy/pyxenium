# pe.py

"""
Portable Executable (PE) Parser
Handles parsing of standard Windows .exe files, which are often found
as the base executable within game dumps.
"""

import struct
import logging
from typing import Optional, NamedTuple

logger = logging.getLogger(__name__)

class PEExecutionInfo(NamedTuple):
    """A simple structure to hold the information needed to start the game."""
    entry_point: int
    base_address: int

class PEParser:
    """A parser for standard Windows PE (Portable Executable) files."""

    def __init__(self, data: bytes):
        self.data = data
        self.file_size = len(data)
        self.is_valid = False
        self.error_message = ""
        self.entry_point = 0
        self.base_address = 0

        try:
            self._parse()
        except Exception as e:
            self.is_valid = False
            self.error_message = f"An unexpected exception occurred during PE parsing: {e}"
            logger.error(self.error_message, exc_info=True)

    def _parse(self):
        """Main parsing logic for PE files."""
        # 1. Check for DOS Header ('MZ')
        if self.file_size < 0x40 or self.data[:2] != b'MZ':
            self.error_message = "Invalid or missing DOS (MZ) header."
            return

        # 2. Find the PE signature location
        pe_location_ptr = 0x3C
        pe_header_location = struct.unpack('<I', self.data[pe_location_ptr : pe_location_ptr + 4])[0]

        # 3. Check for PE Header ('PE\0\0')
        if pe_header_location + 4 > self.file_size or self.data[pe_header_location : pe_header_location + 4] != b'PE\0\0':
            self.error_message = "Invalid or missing PE signature."
            return

        # 4. The Optional Header starts 24 bytes after the PE signature
        optional_header_start = pe_header_location + 24
        
        if optional_header_start + 32 > self.file_size:
            self.error_message = "File too small to contain the PE Optional Header."
            return

        # 5. Check PE magic number for PE32
        pe_magic = struct.unpack('<H', self.data[optional_header_start : optional_header_start + 2])[0]
        if pe_magic != 0x10b:
            self.error_message = f"Unsupported PE format. Expected PE32 (magic 0x10b), got 0x{pe_magic:x}."
            return

        # 6. Extract execution info
        entry_point_rva = struct.unpack('<I', self.data[optional_header_start + 16 : optional_header_start + 20])[0]
        self.base_address = struct.unpack('<I', self.data[optional_header_start + 28 : optional_header_start + 32])[0]
        
        # The final entry point is the base address plus the relative entry point
        self.entry_point = self.base_address + entry_point_rva
        
        self.is_valid = True
        logger.info(f"PE Parsed Successfully: BaseAddress=0x{self.base_address:08X}, EntryPoint=0x{self.entry_point:08X}")

    def get_execution_info(self) -> Optional[PEExecutionInfo]:
        """Returns the essential info needed to start the executable."""
        if not self.is_valid:
            return None
        return PEExecutionInfo(self.entry_point, self.base_address)

    def get_base_file(self) -> Optional[bytes]:
        """For a PE file, the base file is the entire file itself."""
        if not self.is_valid:
            return None
        return self.data
