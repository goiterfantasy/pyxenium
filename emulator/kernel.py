"""
Xbox 360 Kernel Interface
Handles loading and interfacing with the real kernel.exe for authentic syscall handling
"""

import os
import logging
import struct
from typing import Dict, List, Optional, Callable
from pathlib import Path

logger = logging.getLogger(__name__)

class KernelInterface:
    """Interface to Xbox 360 kernel.exe for syscall handling"""
    
    # Xbox 360 System Call Numbers (partial list)
    SYSCALLS = {
        0x0001: 'NtClose',
        0x0002: 'NtCreateFile', 
        0x0003: 'NtReadFile',
        0x0004: 'NtWriteFile',
        0x0005: 'NtQueryInformationFile',
        0x0006: 'NtSetInformationFile',
        0x0007: 'NtQueryDirectoryFile',
        0x0008: 'NtFlushBuffersFile',
        0x0009: 'NtCreateDirectoryObject',
        0x000A: 'NtOpenDirectoryObject',
        0x000B: 'NtQueryDirectoryObject',
        0x000C: 'NtCreateSymbolicLinkObject',
        0x000D: 'NtOpenSymbolicLinkObject',
        0x000E: 'NtQuerySymbolicLinkObject',
        0x000F: 'NtCreateSemaphore',
        0x0010: 'NtReleaseSemaphore',
        0x0011: 'NtCreateMutant',
        0x0012: 'NtReleaseMutant',
        0x0013: 'NtCreateEvent',
        0x0014: 'NtSetEvent',
        0x0015: 'NtPulseEvent',
        0x0016: 'NtClearEvent',
        0x0017: 'NtCreateTimer',
        0x0018: 'NtSetTimer',
        0x0019: 'NtCancelTimer',
        0x001A: 'NtWaitForSingleObjectEx',
        0x001B: 'NtWaitForMultipleObjectsEx',
        0x001C: 'NtSignalAndWaitForSingleObjectEx',
        0x001D: 'NtCreateThread',
        0x001E: 'NtTerminateThread',
        0x001F: 'NtResumeThread',
        0x0020: 'NtSuspendThread',
        0x0021: 'NtGetCurrentProcessId',
        0x0022: 'NtGetCurrentThreadId',
        0x0023: 'NtSetThreadPriority',
        0x0024: 'NtYieldExecution',
        0x0025: 'NtQuerySystemTime',
        0x0026: 'NtQueryPerformanceCounter',
        0x0027: 'NtQueryPerformanceFrequency',
        0x0028: 'NtAllocateVirtualMemory',
        0x0029: 'NtFreeVirtualMemory',
        0x002A: 'NtProtectVirtualMemory',
        0x002B: 'NtQueryVirtualMemory',
        0x002C: 'NtFlushInstructionCache',
        0x002D: 'NtFlushDataCache',
    }
    
    def __init__(self, kernel_path: str, memory_manager, emulator):
        self.kernel_path = kernel_path
        self.memory = memory_manager
        self.emulator = emulator
        self.kernel_loaded = False
        self.kernel_base = 0x80000000
        self.kernel_size = 0
        self.syscall_table = {}
        
        # HLE implementations for syscalls we can't LLE
        self.hle_handlers = self._init_hle_handlers()
        
        # Statistics
        self.syscall_counts = {}
        
    def _init_hle_handlers(self) -> Dict[str, Callable]:
        """Initialize HLE syscall handlers for fallback"""
        return {
            'NtQuerySystemTime': self._hle_query_system_time,
            'NtQueryPerformanceCounter': self._hle_query_performance_counter,
            'NtQueryPerformanceFrequency': self._hle_query_performance_frequency,
            'NtYieldExecution': self._hle_yield_execution,
            'NtAllocateVirtualMemory': self._hle_allocate_virtual_memory,
            'NtFreeVirtualMemory': self._hle_free_virtual_memory,
            'NtGetCurrentProcessId': self._hle_get_current_process_id,
            'NtGetCurrentThreadId': self._hle_get_current_thread_id,
        }
        
    def load_kernel(self) -> bool:
        """Load kernel.exe into memory"""
        if not os.path.exists(self.kernel_path):
            logger.error(f"Kernel file not found: {self.kernel_path}")
            return False
            
        try:
            with open(self.kernel_path, 'rb') as f:
                kernel_data = f.read()
                
            self.kernel_size = len(kernel_data)
            
            # Load kernel into memory at base address
            self.memory.write_memory(self.kernel_base, kernel_data)
            
            # Parse PE header to find entry point and syscall table
            self._parse_kernel_pe(kernel_data)
            
            self.kernel_loaded = True
            logger.info(f"Loaded kernel.exe ({self.kernel_size} bytes) at {self.kernel_base:08X}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load kernel: {e}")
            return False
            
    def _parse_kernel_pe(self, kernel_data: bytes):
        """Parse PE header to extract important information"""
        try:
            # Check PE signature
            if len(kernel_data) < 0x40:
                return
                
            pe_offset = struct.unpack('<I', kernel_data[0x3C:0x40])[0]
            if pe_offset + 4 > len(kernel_data):
                return
                
            pe_sig = kernel_data[pe_offset:pe_offset+4]
            if pe_sig != b'PE\x00\x00':
                logger.warning("Invalid PE signature in kernel.exe")
                return
                
            # Parse COFF header
            coff_header = kernel_data[pe_offset+4:pe_offset+24]
            machine, sections_count = struct.unpack('<HH', coff_header[:4])
            
            logger.debug(f"Kernel PE: machine={machine:04X}, sections={sections_count}")
            
            # In a real implementation, we would:
            # 1. Parse sections and load them at correct virtual addresses
            # 2. Find the syscall table by looking for specific patterns
            # 3. Set up imports/exports tables
            # 4. Handle relocations
            
        except Exception as e:
            logger.warning(f"Failed to parse kernel PE: {e}")
            
    def handle_syscall(self, syscall_num: int, cpu_core) -> bool:
        """Handle system call using LLE or HLE"""
        syscall_name = self.SYSCALLS.get(syscall_num, f"Unknown_{syscall_num:04X}")
        
        # Track syscall usage
        self.syscall_counts[syscall_name] = self.syscall_counts.get(syscall_name, 0) + 1
        
        logger.debug(f"Syscall: {syscall_name} ({syscall_num:04X})")
        
        # Try LLE first if kernel is loaded
        if self.kernel_loaded and self._try_lle_syscall(syscall_num, cpu_core):
            return True
            
        # Fall back to HLE
        if syscall_name in self.hle_handlers:
            return self.hle_handlers[syscall_name](cpu_core)
        else:
            logger.warning(f"Unimplemented syscall: {syscall_name}")
            return False
            
    def _try_lle_syscall(self, syscall_num: int, cpu_core) -> bool:
        """Attempt Low-Level Emulation of syscall using real kernel"""
        # In a real implementation, this would:
        # 1. Look up the syscall address in the kernel's syscall table
        # 2. Set up the CPU state to call the real kernel function
        # 3. Execute the kernel code until it returns
        # 4. Handle any kernel callbacks or interrupts
        
        # For now, we'll simulate this by logging and returning False
        # to fall back to HLE
        logger.debug("LLE syscall not yet implemented, falling back to HLE")
        return False
        
    # HLE Syscall Implementations
    def _hle_query_system_time(self, cpu_core) -> bool:
        """HLE implementation of NtQuerySystemTime"""
        regs = cpu_core.get_current_registers()
        
        # Xbox 360 uses FILETIME format (100ns intervals since 1601)
        # For simplicity, return a fixed time
        filetime = 0x01D7E7F9E7F9E7F9  # Some time in 2021
        
        # Write time to address in r3
        time_ptr = regs.gpr[3]
        if time_ptr != 0:
            self.memory.write_uint32(time_ptr, filetime & 0xFFFFFFFF)
            self.memory.write_uint32(time_ptr + 4, (filetime >> 32) & 0xFFFFFFFF)
            
        # Return STATUS_SUCCESS
        regs.gpr[3] = 0
        return True
        
    def _hle_query_performance_counter(self, cpu_core) -> bool:
        """HLE implementation of NtQueryPerformanceCounter"""
        regs = cpu_core.get_current_registers()
        
        # Return CPU cycle count as performance counter
        counter = self.emulator.cpu.cycles
        
        counter_ptr = regs.gpr[3]
        if counter_ptr != 0:
            self.memory.write_uint32(counter_ptr, counter & 0xFFFFFFFF)
            self.memory.write_uint32(counter_ptr + 4, (counter >> 32) & 0xFFFFFFFF)
            
        regs.gpr[3] = 0
        return True
        
    def _hle_query_performance_frequency(self, cpu_core) -> bool:
        """HLE implementation of NtQueryPerformanceFrequency"""
        regs = cpu_core.get_current_registers()
        
        # Xbox 360 CPU runs at 3.2 GHz
        frequency = 3200000000
        
        freq_ptr = regs.gpr[3]
        if freq_ptr != 0:
            self.memory.write_uint32(freq_ptr, frequency & 0xFFFFFFFF)
            self.memory.write_uint32(freq_ptr + 4, (frequency >> 32) & 0xFFFFFFFF)
            
        regs.gpr[3] = 0
        return True
        
    def _hle_yield_execution(self, cpu_core) -> bool:
        """HLE implementation of NtYieldExecution"""
        # Simply switch to next thread
        cpu_core.switch_thread()
        
        regs = cpu_core.get_current_registers()
        regs.gpr[3] = 0
        return True
        
    def _hle_allocate_virtual_memory(self, cpu_core) -> bool:
        """HLE implementation of NtAllocateVirtualMemory"""
        regs = cpu_core.get_current_registers()
        
        # Parameters (simplified)
        size = regs.gpr[4]  # Size to allocate
        
        # Allocate memory
        addr = self.memory.allocate_memory(size)
        if addr:
            # Write allocated address to pointer in r3
            addr_ptr = regs.gpr[3]
            if addr_ptr != 0:
                self.memory.write_uint32(addr_ptr, addr)
            regs.gpr[3] = 0  # STATUS_SUCCESS
        else:
            regs.gpr[3] = 0xC0000017  # STATUS_NO_MEMORY
            
        return True
        
    def _hle_free_virtual_memory(self, cpu_core) -> bool:
        """HLE implementation of NtFreeVirtualMemory"""
        regs = cpu_core.get_current_registers()
        
        # For now, just return success
        regs.gpr[3] = 0
        return True
        
    def _hle_get_current_process_id(self, cpu_core) -> bool:
        """HLE implementation of NtGetCurrentProcessId"""
        regs = cpu_core.get_current_registers()
        regs.gpr[3] = 1  # Fixed process ID
        return True
        
    def _hle_get_current_thread_id(self, cpu_core) -> bool:
        """HLE implementation of NtGetCurrentThreadId"""
        regs = cpu_core.get_current_registers()
        # Return unique thread ID based on core and thread
        thread_id = (cpu_core.core_id * 2) + cpu_core.current_thread + 1
        regs.gpr[3] = thread_id
        return True
        
    def get_syscall_stats(self) -> str:
        """Get syscall usage statistics"""
        lines = ["Syscall Statistics:"]
        total_calls = sum(self.syscall_counts.values())
        
        for name, count in sorted(self.syscall_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_calls) * 100 if total_calls > 0 else 0
            lines.append(f"  {name}: {count} ({percentage:.1f}%)")
            
        lines.append(f"Total syscalls: {total_calls}")
        return '\n'.join(lines)