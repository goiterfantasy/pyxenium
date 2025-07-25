# cpu.py

"""
Xbox 360 Xenon CPU Emulation
This final version includes a critical fix for all D-form memory instructions
(lwz, stw, stwu, etc.) to correctly handle the ra=0 addressing case,
preventing stack corruption and ensuring stable execution.
"""

import logging
import struct
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class CPUException(Exception):
    pass

@dataclass
class CPURegisters:
    gpr: List[int]
    fpr: List[float]
    pc: int
    lr: int
    ctr: int
    cr: int
    xer: int
    msr: int
    
    def __init__(self):
        self.gpr = [0] * 32
        self.fpr = [0.0] * 32
        self.pc = 0
        self.lr = 0
        self.ctr = 0
        self.cr = 0
        self.xer = 0
        self.msr = 0x8000

class XenonCore:
    """Single Xenon CPU core (supports 2 hardware threads)"""
    
    def __init__(self, core_id: int, memory_manager):
        self.core_id = core_id
        self.memory = memory_manager
        self.threads = [CPURegisters(), CPURegisters()]
        self.current_thread = 0
        self.running = False
        self.instruction_handlers = self._init_instruction_handlers()
        
    def _init_instruction_handlers(self) -> Dict[int, Callable]:
        """Initialize instruction handler lookup table."""
        return {
            0x0A: self._handle_cmpli,
            0x0B: self._handle_cmpwi,
            0x0E: self._handle_addi,
            0x0F: self._handle_addis,
            0x10: self._handle_bc,
            0x12: self._handle_b,
            0x13: self._handle_bclr_or_blr,
            0x15: self._handle_rlwinm,
            0x18: self._handle_ori,
            0x1A: self._handle_xori,
            0x1C: self._handle_andi,
            0x1F: self._handle_extended_opcode,
            0x20: self._handle_lwz,
            0x24: self._handle_stw,
            0x25: self._handle_stwu,
            0x26: self._handle_stb,
            0x3A: self._handle_stwux,
            0x3E: self._handle_stfd,
        }
        
    def get_current_registers(self) -> CPURegisters:
        return self.threads[self.current_thread]
        
    def decode_and_execute(self, instruction: int):
        if instruction == 0x60000000: # Explicit NOP
            return # Do nothing
        
        opcode = (instruction >> 26) & 0x3F
        if opcode in self.instruction_handlers:
            self.instruction_handlers[opcode](instruction)
        else:
            logger.warning(f"Unimplemented opcode: {opcode} for instruction {instruction:08X}")

    # --- INSTRUCTION HANDLERS ---
    def _get_effective_address_x_form(self, ra, rb):
        regs = self.get_current_registers()
        addr_a = regs.gpr[ra] if ra != 0 else 0
        addr_b = regs.gpr[rb]
        return (addr_a + addr_b) & 0xFFFFFFFF
    def _handle_addi(self, instruction):
        rt, ra, imm = (instruction >> 21) & 0x1F, (instruction >> 16) & 0x1F, instruction & 0xFFFF
        if imm & 0x8000: imm -= 0x10000
        regs = self.get_current_registers()
        if ra == 0: regs.gpr[rt] = imm & 0xFFFFFFFF
        else: regs.gpr[rt] = (regs.gpr[ra] + imm) & 0xFFFFFFFF
    def _handle_addis(self, instruction):
        rt, ra, imm = (instruction >> 21) & 0x1F, (instruction >> 16) & 0x1F, (instruction & 0xFFFF) << 16
        regs = self.get_current_registers()
        if ra == 0: regs.gpr[rt] = imm
        else: regs.gpr[rt] = (regs.gpr[ra] + imm) & 0xFFFFFFFF
    def _handle_ori(self, instruction):
        rs, ra, imm = (instruction >> 21) & 0x1F, (instruction >> 16) & 0x1F, instruction & 0xFFFF
        regs = self.get_current_registers()
        regs.gpr[ra] = regs.gpr[rs] | imm
    def _handle_xori(self, instruction):
        rs, ra, imm = (instruction >> 21) & 0x1F, (instruction >> 16) & 0x1F, instruction & 0xFFFF
        regs = self.get_current_registers()
        regs.gpr[ra] = regs.gpr[rs] ^ imm
    def _handle_andi(self, instruction):
        rs, ra, imm = (instruction >> 21) & 0x1F, (instruction >> 16) & 0x1F, instruction & 0xFFFF
        regs = self.get_current_registers()
        regs.gpr[ra] = regs.gpr[rs] & imm
    def _handle_lwz(self, instruction):
        rt, ra, offset = (instruction >> 21) & 0x1F, (instruction >> 16) & 0x1F, instruction & 0xFFFF
        if offset & 0x8000: offset -= 0x10000
        regs = self.get_current_registers()
        base = regs.gpr[ra] if ra != 0 else 0
        addr = (base + offset) & 0xFFFFFFFF
        regs.gpr[rt] = self.memory.read_uint32(addr)
    def _handle_stw(self, instruction):
        rs, ra, offset = (instruction >> 21) & 0x1F, (instruction >> 16) & 0x1F, instruction & 0xFFFF
        if offset & 0x8000: offset -= 0x10000
        regs = self.get_current_registers()
        base = regs.gpr[ra] if ra != 0 else 0
        addr = (base + offset) & 0xFFFFFFFF
        self.memory.write_uint32(addr, regs.gpr[rs])
    def _handle_stwu(self, instruction):
        rs, ra, offset = (instruction >> 21) & 0x1F, (instruction >> 16) & 0x1F, instruction & 0xFFFF
        if offset & 0x8000: offset -= 0x10000
        regs = self.get_current_registers()
        # The ra=0 case for stwu is technically an invalid instruction form,
        # but we handle it defensively as if it's a normal store.
        base = regs.gpr[ra] if ra != 0 else 0
        addr = (base + offset) & 0xFFFFFFFF
        self.memory.write_uint32(addr, regs.gpr[rs])
        if ra != 0:
            regs.gpr[ra] = addr
    def _handle_stb(self, instruction):
        rs, ra, offset = (instruction >> 21) & 0x1F, (instruction >> 16) & 0x1F, instruction & 0xFFFF
        if offset & 0x8000: offset -= 0x10000
        regs = self.get_current_registers()
        base = regs.gpr[ra] if ra != 0 else 0
        addr = (base + offset) & 0xFFFFFFFF
        self.memory.write_uint8(addr, regs.gpr[rs] & 0xFF)
    def _handle_stfd(self, instruction):
        rs, ra, offset = (instruction >> 21) & 0x1F, (instruction >> 16) & 0x1F, instruction & 0xFFFF
        if offset & 0x8000: offset -= 0x10000
        regs = self.get_current_registers()
        base = regs.gpr[ra] if ra != 0 else 0
        addr = (base + offset) & 0xFFFFFFFF
        data = struct.pack('>d', regs.fpr[rs])
        self.memory.write_memory(addr, data)
    def _handle_stwux(self, instruction):
        rs, ra, rb = (instruction >> 21) & 0x1F, (instruction >> 16) & 0x1F, (instruction >> 11) & 0x1F
        regs = self.get_current_registers()
        addr = (regs.gpr[ra] + regs.gpr[rb]) & 0xFFFFFFFF
        self.memory.write_uint32(addr, regs.gpr[rs])
        regs.gpr[ra] = addr
    def _handle_b(self, instruction):
        target = instruction & 0x3FFFFFC
        if target & 0x2000000: target -= 0x4000000
        regs = self.get_current_registers()
        if instruction & 1: regs.lr = (regs.pc + 4) & 0xFFFFFFFF
        if instruction & 2: regs.pc = target & 0xFFFFFFFF
        else: regs.pc = (regs.pc + target) & 0xFFFFFFFF
    def _handle_bc(self, instruction):
        bo, bi, target = (instruction >> 21) & 0x1F, (instruction >> 16) & 0x1F, instruction & 0xFFFC
        if target & 0x8000: target -= 0x10000
        regs = self.get_current_registers()
        ctr_ok = ((bo >> 2) & 1) or (((regs.ctr - 1) != 0) ^ ((bo >> 1) & 1))
        if not ((bo >> 2) & 1): regs.ctr = (regs.ctr - 1) & 0xFFFFFFFF
        cond_ok = ((bo >> 4) & 1) or (((regs.cr >> (31 - bi)) & 1) == ((bo >> 3) & 1))
        if ctr_ok and cond_ok:
            if instruction & 1: regs.lr = (regs.pc + 4) & 0xFFFFFFFF
            if instruction & 2: regs.pc = target & 0xFFFFFFFF
            else: regs.pc = (regs.pc + target) & 0xFFFFFFFF
        else: regs.pc += 4
    def _handle_bclr_or_blr(self, instruction):
        regs = self.get_current_registers()
        regs.pc = regs.lr & 0xFFFFFFFC
    def _update_cr_field(self, field_idx: int, value: int):
        regs = self.get_current_registers()
        shift = (7 - field_idx) * 4
        mask = ~(0xF << shift)
        regs.cr = (regs.cr & mask) | (value << shift)
    def _handle_cmpli(self, instruction):
        crfD, ra, imm = (instruction >> 23) & 0x7, (instruction >> 16) & 0x1F, instruction & 0xFFFF
        val_a = self.get_current_registers().gpr[ra]
        result = 0
        if val_a < imm: result |= 0b1000
        elif val_a > imm: result |= 0b0100
        else: result |= 0b0010
        self._update_cr_field(crfD, result)
    def _handle_cmpwi(self, instruction):
        crfD, ra, imm = (instruction >> 23) & 0x7, (instruction >> 16) & 0x1F, instruction & 0xFFFF
        if imm & 0x8000: imm -= 0x10000
        regs = self.get_current_registers()
        val_a = regs.gpr[ra]
        if val_a & 0x80000000: val_a -= 0x100000000
        result = 0
        if val_a < imm: result |= 0b1000
        elif val_a > imm: result |= 0b0100
        else: result |= 0b0010
        self._update_cr_field(crfD, result)
    def _handle_rlwinm(self, instruction):
        rs, ra, sh, mb, me = (instruction >> 21) & 0x1F, (instruction >> 16) & 0x1F, (instruction >> 11) & 0x1F, (instruction >> 6) & 0x1F, (instruction >> 1) & 0x1F
        regs = self.get_current_registers()
        val = regs.gpr[rs]
        rotated = ((val << sh) | (val >> (32 - sh))) & 0xFFFFFFFF
        mask = ((1 << (31 - mb + 1)) - 1) ^ ((1 << (31 - me)) - 1)
        regs.gpr[ra] = rotated & mask
        if instruction & 1:
            cr_val = 0
            if regs.gpr[ra] < 0: cr_val |= 0b1000
            elif regs.gpr[ra] > 0: cr_val |= 0b0100
            else: cr_val |= 0b0010
            self._update_cr_field(0, cr_val)
    def _handle_extended_opcode(self, instruction):
        ext_opcode = (instruction >> 1) & 0x3FF
        handlers = {
            0: self._handle_cmp,
            266: self._handle_add,
            339: self._handle_mfspr,
            444: self._handle_or,
            467: self._handle_mtspr,
            1023: self._handle_dcbz,
        }
        if ext_opcode in handlers:
            handlers[ext_opcode](instruction)
        else:
            logger.warning(f"Unimplemented extended opcode: {ext_opcode} in instruction {instruction:08X}")
    def _handle_cmp(self, instruction):
        crfD, ra, rb = (instruction >> 23) & 0x7, (instruction >> 16) & 0x1F, (instruction >> 11) & 0x1F
        regs = self.get_current_registers()
        val_a = regs.gpr[ra]
        val_b = regs.gpr[rb]
        if val_a & 0x80000000: val_a -= 0x100000000
        if val_b & 0x80000000: val_b -= 0x100000000
        result = 0
        if val_a < val_b: result |= 0b1000
        elif val_a > val_b: result |= 0b0100
        else: result |= 0b0010
        self._update_cr_field(crfD, result)
    def _handle_add(self, instruction):
        rt, ra, rb = (instruction >> 21) & 0x1F, (instruction >> 16) & 0x1F, (instruction >> 11) & 0x1F
        regs = self.get_current_registers()
        result = (regs.gpr[ra] + regs.gpr[rb]) & 0xFFFFFFFF
        regs.gpr[rt] = result
        if instruction & 1:
            cr_val = 0
            if result < 0: cr_val |= 0b1000
            elif result > 0: cr_val |= 0b0100
            else: cr_val |= 0b0010
            self._update_cr_field(0, cr_val)
    def _handle_or(self, instruction):
        rs, ra, rb = (instruction >> 21) & 0x1F, (instruction >> 16) & 0x1F, (instruction >> 11) & 0x1F
        regs = self.get_current_registers()
        result = regs.gpr[rs] | regs.gpr[rb]
        regs.gpr[ra] = result
        if instruction & 1:
            cr_val = 0
            if result < 0: cr_val |= 0b1000
            elif result > 0: cr_val |= 0b0100
            else: cr_val |= 0b0010
            self._update_cr_field(0, cr_val)
    def _get_spr_num(self, instruction):
        spr_field = (instruction >> 11) & 0x3FF
        return ((spr_field & 0x1F) << 5) | ((spr_field >> 5) & 0x1F)
    def _handle_mtspr(self, instruction):
        rs, spr_num = (instruction >> 21) & 0x1F, self._get_spr_num(instruction)
        regs = self.get_current_registers()
        value = regs.gpr[rs]
        if spr_num == 8: regs.lr = value
        elif spr_num == 9: regs.ctr = value
        else: logger.warning(f"Write to unimplemented SPR: {spr_num}")
    def _handle_mfspr(self, instruction):
        rt, spr_num = (instruction >> 21) & 0x1F, self._get_spr_num(instruction)
        regs = self.get_current_registers()
        if spr_num == 8: regs.gpr[rt] = regs.lr
        elif spr_num == 9: regs.gpr[rt] = regs.ctr
        else:
            logger.warning(f"Read from unimplemented SPR: {spr_num}")
            regs.gpr[rt] = 0
    def _handle_dcbz(self, instruction):
        ra, rb = (instruction >> 16) & 0x1F, (instruction >> 11) & 0x1F
        addr = self._get_effective_address_x_form(ra, rb)
        self.memory.write_memory(addr & ~127, bytearray(128))
    def step(self) -> bool:
        try:
            regs = self.get_current_registers()
            instruction = self.memory.read_uint32(regs.pc)
            old_pc = regs.pc
            self.decode_and_execute(instruction)
            if regs.pc == old_pc: regs.pc += 4
            return True
        except Exception as e:
            logger.error(f"CPU execution error at PC=0x{self.get_current_registers().pc:08X}: {e}", exc_info=True)
            self.running = False
            return False

class XenonCPU:
    """Xbox 360 Xenon CPU - 3 cores, 6 hardware threads total"""
    
    def __init__(self, memory_manager):
        self.memory = memory_manager
        self.cores = [XenonCore(i, memory_manager) for i in range(3)]
        self.current_core_idx = 0
        self.running = False
        logger.info("Initialized Xenon CPU with 3 cores, 6 hardware threads")
        
    def reset(self):
        for core in self.cores:
            for thread_regs in core.threads:
                thread_regs.pc = 0
        self.running = False
        logger.info("CPU reset to initial state")
        
    def run(self, max_cycles: int = 1000000):
        self.running = True
        for _ in range(max_cycles):
            if not self.running: break
            core = self.cores[self.current_core_idx]
            if not core.step():
                self.running = False
                break
            if max_cycles > 1:
                self.current_core_idx = (self.current_core_idx + 1) % 3
        
    def stop(self):
        self.running = False

    def get_current_core(self) -> XenonCore:
        """Get the currently active core."""
        return self.cores[self.current_core_idx]

    def get_register_dump(self) -> str:
        """Get formatted register dump for debugging."""
        core = self.get_current_core()
        regs = core.get_current_registers()
        lines = [f"--- Core {self.current_core_idx}, Thread {core.current_thread} ---"]
        lines.append(f"PC: {regs.pc:08X}  LR: {regs.lr:08X}  CTR: {regs.ctr:08X}  CR: {regs.cr:08X}")
        for i in range(0, 32, 4):
            line = " ".join([f"r{i+j:<2}={regs.gpr[i+j]:08X}" for j in range(4)])
            lines.append(line)
        return '\n'.join(lines)
