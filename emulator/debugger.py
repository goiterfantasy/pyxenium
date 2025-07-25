# debugger.py

"""
Xbox 360 Emulator Debugger
This version features a much more capable PowerPC disassembler to produce
human-readable assembly code.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class PowerPCDisassembler:
    """A more capable PowerPC disassembler for debugging."""

    @staticmethod
    def disassemble_instruction(address: int, instr: int) -> str:
        opcode = (instr >> 26) & 0x3F

        # I-FORM (Branch immediate)
        if opcode == 0x12: # b
            li = instr & 0x03FFFFFC
            if li & 0x02000000: li -= 0x04000000
            aa = "a" if (instr & 2) else ""
            lk = "l" if (instr & 1) else ""
            target = li if aa else address + li
            return f"b{lk}{aa}".ljust(8) + f"0x{target:08X}"

        # B-FORM (Branch conditional)
        if opcode == 0x10: # bc
            bo, bi = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            bd = instr & 0xFFFC
            if bd & 0x8000: bd -= 0x10000
            aa = "a" if (instr & 2) else ""
            lk = "l" if (instr & 1) else ""
            target = bd if aa else address + bd
            # Simplified mnemonic for common cases
            if bo == 4 and bi == 2: mnemonic = "bne"
            elif bo == 12 and bi == 2: mnemonic = "beq"
            else: mnemonic = f"bc{lk}{aa}"
            return mnemonic.ljust(8) + f"cr{bi//4}, 0x{target:08X}"
        
        # D-FORM (Immediate arithmetic, loads/stores)
        if opcode in [0x0E, 0x0F, 0x20, 0x24, 0x25]:
            rt, ra = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            imm = instr & 0xFFFF
            if imm & 0x8000: imm -= 0x10000
            
            mnemonics = {0x0E: "addi", 0x0F: "addis", 0x20: "lwz", 0x24: "stw", 0x25: "stwu"}
            mnemonic = mnemonics[opcode]

            # Common alias li/lis
            if opcode == 0x0E and ra == 0: return f"li".ljust(8) + f"r{rt}, {imm}"
            if opcode == 0x0F and ra == 0: return f"lis".ljust(8) + f"r{rt}, {imm >> 16}"

            if opcode in [0x20, 0x24, 0x25]: # Load/Store format
                return mnemonic.ljust(8) + f"r{rt}, {imm}(r{ra})"
            else: # Arithmetic format
                return mnemonic.ljust(8) + f"r{rt}, r{ra}, {imm}"
        
        # XL-FORM (Branch to LR/CTR)
        if opcode == 0x13: # bclr/blr
            bo, bi = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            if bo == 20 and bi == 0: return "blr".ljust(8)
            return "bclr".ljust(8) + f"{bo}, {bi}"

        # XFX-FORM (Move to/from SPR)
        if opcode == 0x1F:
            rt, ra, rb = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F, (instr >> 11) & 0x1F
            ext = (instr >> 1) & 0x3FF
            
            if ext == 339: # mfspr
                spr = ((rb & 0x1F) << 5) | ((ra >> 5) & 0x1F)
                if spr == 8: spr_name = "lr"
                elif spr == 9: spr_name = "ctr"
                else: spr_name = f"{spr}"
                return "mfspr".ljust(8) + f"r{rt}, {spr_name}"
            
            if ext == 467: # mtspr
                spr = ((rb & 0x1F) << 5) | ((ra >> 5) & 0x1F)
                if spr == 8: spr_name = "lr"
                elif spr == 9: spr_name = "ctr"
                else: spr_name = f"{spr}"
                return "mtspr".ljust(8) + f"{spr_name}, r{rt}"

        return f"op={opcode:02X}".ljust(8) + f"raw=0x{instr:08X}"

class EmulatorDebugger:
    """Xbox 360 Emulator Debugger"""
    
    def __init__(self, emulator):
        self.emulator = emulator
        
    def step_instruction(self) -> bool:
        """Execute a single instruction."""
        if not self.emulator.running:
            self.emulator.cpu.running = True
        success = self.emulator.cpu.run(1)
        self.emulator.cpu.running = False
        if success:
            pc = self.emulator.cpu.get_current_core().get_current_registers().pc
            logger.info(f"Stepped one instruction. New PC: 0x{pc:08X}")
        return success

    def dump_registers(self) -> str:
        """Get formatted register dump."""
        return self.emulator.cpu.get_register_dump()
        
    def disassemble_range(self, start_addr: int, count: int = 10) -> List[str]:
        """Disassemble instructions in a range."""
        lines = []
        current_pc = self.emulator.cpu.get_current_core().get_current_registers().pc
        
        for i in range(count):
            addr = start_addr + (i * 4)
            try:
                instruction = self.emulator.memory.read_uint32(addr)
                disasm = PowerPCDisassembler.disassemble_instruction(addr, instruction)
                
                marker = " -> " if addr == current_pc else "    "
                lines.append(f"{marker}0x{addr:08X}:  {disasm}")
            except Exception as e:
                lines.append(f"    0x{addr:08X}: <error: {e}>")
                break
        return lines

    def print_help(self) -> str:
        """Get debugger help text."""
        return """
--- Xbox 360 Emulator Debugger Commands ---
  start         - Start or resume emulation.
  stop          - Pause emulation.
  step          - Execute a single CPU instruction.
  regs          - Show current CPU registers.
  disasm [addr] - Disassemble code at [addr] or current PC.
  mem <addr>    - Dump 256 bytes of memory starting at <addr>.
  quit          - Exit the emulator.
"""
