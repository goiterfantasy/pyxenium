# filesystem.py

"""
Xbox 360 File System Emulation
Handles file system operations and device mounting
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, BinaryIO
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FileHandle:
    """Represents an open file handle"""
    handle_id: int
    path: str
    mode: str
    position: int
    file_obj: Optional[BinaryIO] = None

class XboxDevice:
    """Represents an Xbox 360 storage device"""
    
    def __init__(self, name: str, mount_path: Path, device_type: str = "HDD"):
        self.name = name
        self.mount_path = Path(mount_path)
        self.device_type = device_type
        self.mounted = False
        self.mount_path.mkdir(parents=True, exist_ok=True)
        
    def mount(self) -> bool:
        """Mount the device"""
        self.mounted = True
        logger.info(f"Mounted {self.device_type} device '{self.name}' at {self.mount_path}")
        return True

class FileSystem:
    """Xbox 360 File System Emulator"""
    
    DEVICE_MAPPINGS = {
        'game:': 'Game',
        'hdd1:': 'HDD',
        'dvd:': 'DVD',
    }
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.devices: Dict[str, XboxDevice] = {}
        self.file_handles: Dict[int, FileHandle] = {}
        self.next_handle_id = 1
        
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._init_devices()
        
    def _init_devices(self):
        """Initialize Xbox 360 storage devices"""
        for prefix, name in self.DEVICE_MAPPINGS.items():
            device_path = self.base_path / name.lower()
            device = XboxDevice(name, device_path, name)
            device.mount()
            self.devices[name] = device
        logger.info(f"Initialized {len(self.devices)} storage devices")
        
    def resolve_path(self, xbox_path: str) -> Optional[Path]:
        """Resolve Xbox path to host file system path"""
        xbox_path = xbox_path.replace('\\', '/')
        
        for prefix, dev_name in self.DEVICE_MAPPINGS.items():
            if xbox_path.lower().startswith(prefix.lower()):
                device = self.devices.get(dev_name)
                if device and device.mounted:
                    relative_path = xbox_path[len(prefix):].lstrip('/')
                    return device.mount_path / relative_path
                
        logger.warning(f"Could not resolve path: {xbox_path}")
        return None
        
    def open_file(self, path: str, mode: str = 'rb') -> Optional[int]:
        """Open a file and return handle ID"""
        host_path = self.resolve_path(path)
        if not host_path:
            return None
        try:
            host_path.parent.mkdir(parents=True, exist_ok=True)
            file_obj = open(host_path, mode)
            handle_id = self.next_handle_id
            self.next_handle_id += 1
            
            handle = FileHandle(handle_id, str(host_path), mode, 0, file_obj)
            self.file_handles[handle_id] = handle
            logger.debug(f"Opened file: {path} -> {host_path} (handle {handle_id})")
            return handle_id
        except Exception as e:
            logger.error(f"Failed to open file {path}: {e}")
            return None
            
    def read_file(self, handle_id: int, size: int) -> Optional[bytes]:
        """Read data from file handle"""
        handle = self.file_handles.get(handle_id)
        if handle and handle.file_obj:
            data = handle.file_obj.read(size)
            handle.position += len(data)
            return data
        return None
            
    def close_file(self, handle_id: int) -> bool:
        """Close file handle"""
        if handle_id in self.file_handles:
            handle = self.file_handles.pop(handle_id)
            if handle.file_obj:
                handle.file_obj.close()
            logger.debug(f"Closed file handle {handle_id}")
            return True
        return False

    def file_exists(self, path: str) -> bool:
        """Check if file exists"""
        host_path = self.resolve_path(path)
        return host_path is not None and host_path.exists()
