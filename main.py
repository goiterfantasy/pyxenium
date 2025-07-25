# main.py

"""
Xbox 360 Hybrid HLE-LLE Emulator
Main entry point and CLI interface with enhanced debugging commands.
"""

import os
import sys
import logging
import argparse
import time
from pathlib import Path

# --- Logger setup moved to global scope ---
logger = logging.getLogger(__name__)

# Add emulator package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import EmulatorConfig
from emulator import Xbox360Emulator

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('emulator.log', mode='w')
        ]
    )

def interactive_mode(emulator: Xbox360Emulator):
    """Interactive debugging mode"""
    print("\nXbox 360 Emulator Interactive Mode")
    print("Type 'help' for available commands.")
    
    debugger = emulator.debugger
    
    while True:
        try:
            cmd_line = input("(xbox360-emu) ").strip()
            if not cmd_line: continue
            
            cmd = cmd_line.split()
            command = cmd[0].lower()
            
            if command in ['quit', 'exit', 'q']:
                if emulator.running: emulator.stop()
                break
            elif command == 'help':
                print(debugger.print_help())
            elif command == 'start':
                if not emulator.running:
                    emulator.start()
                    print("Emulation started in background. Use 'stop' to halt.")
                else:
                    print("Emulation is already running.")
            elif command == 'stop':
                if emulator.running:
                    emulator.stop()
                    print("Emulation paused.")
                else:
                    print("Emulation is not running.")
            elif command == 'step':
                debugger.step_instruction()
                print(debugger.dump_registers())
                pc = emulator.cpu.cores[emulator.cpu.current_core_idx].get_current_registers().pc
                print("\nDisassembly:")
                print("\n".join(debugger.disassemble_range(pc)))
            elif command == 'regs':
                print(debugger.dump_registers())
            elif command == 'disasm':
                addr_str = cmd[1] if len(cmd) > 1 else "pc"
                pc = emulator.cpu.cores[emulator.cpu.current_core_idx].get_current_registers().pc
                addr = int(addr_str, 16) if addr_str != "pc" else pc
                print("\n".join(debugger.disassemble_range(addr)))
            elif command == 'mem':
                if len(cmd) < 2:
                    print("Usage: mem <hex_address>")
                    continue
                addr = int(cmd[1], 16)
                print(f"Memory dump at 0x{addr:08X} not fully implemented yet.")
            else:
                print(f"Unknown command: {command}")
                
        except KeyboardInterrupt:
            print("\nUse 'quit' or 'exit' to leave.")
        except Exception as e:
            # This will now work correctly
            logger.error(f"Error in command: {e}", exc_info=True)
            
    print("Exiting interactive mode.")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Xbox 360 Hybrid HLE-LLE Emulator')
    parser.add_argument('--config', '-c', default='config.json', help='Configuration file')
    parser.add_argument('--game', '-g', help='Game executable to load')
    parser.add_argument('--kernel', help='Path to kernel.exe')
    
    args = parser.parse_args()
    
    setup_logging()
    
    logger.info("Starting Xbox 360 Emulator")
    
    try:
        config = EmulatorConfig(args.config)
        if args.kernel:
            config.set('system.kernel_path', args.kernel)
            
        emulator = Xbox360Emulator(config.config)
        emulator.reset()
        
        if not args.game:
            logger.error("No game specified. Please use the --game argument.")
            interactive_mode(emulator)
            return 0

        if not emulator.load_game(args.game):
            logger.error(f"Failed to load game: {args.game}. Check logs for details.")
            return 1
            
        interactive_mode(emulator)
        
    except Exception as e:
        logger.error(f"A fatal error occurred: {e}", exc_info=True)
        return 1
        
    return 0

if __name__ == '__main__':
    sys.exit(main())
