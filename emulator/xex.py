# xex.py

"""
Xbox 360 Executable (XEX) Parser
This version correctly detects compressed XEX files, which is a common
format for retail games.
"""

import struct
import logging
from typing import Optional, Dict, NamedTuple

logger = logging.getLogger(__name__)

# --- XEX Header Keys ---
XEX_HEADER_EXECUTION_ID = 0x00040006
XEX_HEADER_BASE_FILE_INFO = 0x00020001

class XEXExecutionInfo(NamedTuple):
    entry_point: int
    base_address: int

class XEXParser:
    """A robust parser that can now detect XEX compression."""

    def __init__(self, data: bytes):
        self.data = data
        self.file_size = len(data)
        self.is_valid = False
        self.error_message = ""
        self.entry_point = 0
        self.base_address = 0
        self.exe_offset = 0
        self.descriptors: Dict[int, int] = {}

        try:
            self._parse()
        except Exception as e:
            self.is_valid = False
            self.error_message = f"An unexpected exception occurred during parsing: {e}"
            logger.error(self.error_message, exc_info=True)

    def _parse(self):
        """Main parsing logic with compression detection."""
        if self.file_size < 24 or self.data[:4] != b'XEX2':
            self.error_message = "Invalid XEX2 magic or file too small."
            return

        # --- Parse XEX Header and Directory ---
        _, _, self.exe_offset, _, _, entry_count = struct.unpack('>IIIIII', self.data[:24])
        
        dir_offset = 24
        for _ in range(entry_count):
            if dir_offset + 8 > self.file_size:
                self.error_message = "Header directory is corrupt or truncated."
                return
            key, offset = struct.unpack('>II', self.data[dir_offset:dir_offset+8])
            self.descriptors[key] = offset
            dir_offset += 8

        # --- Check for Compression FIRST ---
        base_info_offset = self.descriptors.get(XEX_HEADER_BASE_FILE_INFO)
        if not base_info_offset:
            self.error_message = "XEX is missing mandatory Base File Info descriptor."
            return

        if base_info_offset + 12 > self.file_size:
            self.error_message = "File too small to read Base File Info descriptor."
            return
            
        compression_info, _, _ = struct.unpack('>III', self.data[base_info_offset : base_info_offset+12])
        
        # The compression type is stored in the top 4 bits.
        compression_type = (compression_info >> 28) & 0xF
        if compression_type != 0:
            # 1 = basic compression, 2 = LZX compression
            self.error_message = "XEX file is compressed, which is not yet supported by this emulator."
            logger.error(self.error_message)
            # We cannot proceed if the file is compressed.
            return

        # --- If NOT compressed, proceed with PE parsing ---
        
        # Get Entry Point
        exec_id_offset = self.descriptors.get(XEX_HEADER_EXECUTION_ID)
        if not exec_id_offset or exec_id_offset + 8 > self.file_size:
            self.error_message = "Could not find or read entry point descriptor."
            return
        self.entry_point = struct.unpack('>I', self.data[exec_id_offset+4 : exec_id_offset+8])[0]

        # Get Base Address from the embedded PE header
        pe_header_start = self.exe_offset
        if pe_header_start <= 0 or pe_header_start + 0x40 > self.file_size:
            self.error_message = f"Invalid PE header offset: 0x{pe_header_start:X}"
            return

        pe_sig_offset_ptr = pe_header_start + 0x3C
        pe_sig_offset = struct.unpack('<I', self.data[pe_sig_offset_ptr : pe_sig_offset_ptr+4])[0]
        optional_header_start = pe_header_start + pe_sig_offset + 24
        
        if optional_header_start + 32 > self.file_size:
            self.error_message = "Calculated Optional Header start is out of file bounds."
            return

        self.base_address = struct.unpack('<I', self.data[optional_header_start + 28 : optional_header_start + 32])[0]
        
        self.is_valid = True
        logger.info(f"Uncompressed XEX Parsed: BaseAddress=0x{self.base_address:08X}, EntryPoint=0x{self.entry_point:08X}")

    def get_execution_info(self) -> Optional[XEXExecutionInfo]:
        if not self.is_valid: return None
        return XEXExecutionInfo(self.entry_point, self.base_address)

    def get_base_file(self) -> Optional[bytes]:
        if not self.is_valid: return None
        return self.data[self.exe_offset:]
