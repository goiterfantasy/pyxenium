# __init__.py

"""
Xbox 360 Emulator Package
"""

from .emulator import Xbox360Emulator
from .cpu import XenonCPU
from .memory import MemoryManager
from .kernel import KernelInterface
from .gpu import XenosGPU
from .filesystem import FileSystem
from .xex import XEXParser
from .pe import PEParser # Added the new PE parser

__all__ = [
    'Xbox360Emulator',
    'XenonCPU', 
    'MemoryManager',
    'KernelInterface',
    'XenosGPU',
    'FileSystem',
    'XEXParser',
    'PEParser' # Export the new PE parser
]
