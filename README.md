# Xbox 360 Hybrid HLE-LLE Emulator

A Python 3 implementation of an Xbox 360 emulator using a hybrid High-Level Emulation (HLE) and Low-Level Emulation (LLE) approach, with authentic syscall handling through the real kernel.exe.

## Features

### Core Emulation
- **PowerPC Xenon CPU**: Full 3-core, 6-thread processor emulation
- **Memory Management**: Authentic Xbox 360 memory layout with virtual addressing
- **Xenos GPU**: Software-based graphics processing with basic rendering pipeline
- **File System**: Complete Xbox 360 file system emulation with device mounting
- **Kernel Interface**: Real kernel.exe integration for authentic syscall handling

### Hybrid Architecture
- **HLE Components**: High-level emulation for performance-critical operations
- **LLE Components**: Low-level emulation using real Xbox 360 kernel for accuracy
- **Syscall Passthrough**: Routes system calls to authentic kernel.exe when available
- **Fallback System**: HLE implementations for unsupported LLE operations

### Development Features
- **Interactive Debugger**: Full debugging support with breakpoints and tracing
- **Disassembler**: PowerPC instruction disassembly and analysis
- **Memory Inspector**: Comprehensive memory viewing and editing
- **Performance Profiling**: Detailed execution statistics and timing analysis

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Obtain Xbox 360 kernel.exe (not included - must be legally obtained)

3. Create directory structure:
```bash
mkdir -p games saves cache
```

## Usage

### Basic Usage
```bash
# Start emulator with default settings
python main.py

# Load a game
python main.py --game path/to/game.xex

# Interactive debugging mode
python main.py --interactive

# Run automated tests
python main.py --test
```

### Configuration
Edit `config.yaml` to customize emulator settings:

```yaml
system:
  kernel_path: 'kernel.exe'
  memory_size: 536870912  # 512MB
  cpu_threads: 6

emulation:
  cpu_mode: 'interpreter'
  gpu_backend: 'software'
  hle_kernel: false  # Use real kernel.exe

debug:
  enable_logging: true
  log_level: 'INFO'
  trace_execution: false
```

### Interactive Commands

In interactive mode, use these commands:

- `help` - Show all available commands
- `load <game>` - Load game executable
- `start/stop/pause/resume` - Control emulation
- `step` - Execute single instruction
- `regs` - Show CPU registers
- `disasm [addr] [count]` - Disassemble instructions
- `mem <addr> [size]` - Dump memory contents
- `bp <addr>` - Set breakpoint
- `trace on/off` - Enable/disable tracing
- `stats` - Show performance statistics
- `screenshot [filename]` - Take screenshot

## Architecture

### CPU Emulation (XenonCPU)
- PowerPC instruction set implementation
- 3 cores with 2 hardware threads each
- Register management and context switching
- Branch prediction and pipeline simulation

### Memory Management (MemoryManager)
- Xbox 360 memory layout emulation
- Virtual-to-physical address translation
- Memory protection and access control
- Efficient read/write operations

### Kernel Interface (KernelInterface)
- Real kernel.exe loading and execution
- Syscall table parsing and routing
- HLE fallback implementations
- Statistics tracking and debugging

### GPU Emulation (XenosGPU)
- Command buffer processing
- Software rasterization
- Basic shader emulation
- Frame buffer management

### File System (FileSystem)
- Xbox 360 device mounting (HDD, DVD, USB, etc.)
- Path resolution and translation
- File I/O operations
- Game loading support

## Development

### Adding New Instructions
1. Add opcode to `PowerPCOpcode` enum in `cpu.py`
2. Implement handler method in `XenonCore`
3. Add to instruction handler lookup table
4. Update disassembler in `debugger.py`

### Adding New Syscalls
1. Add syscall number to `SYSCALLS` dict in `kernel.py`
2. Implement HLE handler method
3. Add to HLE handlers lookup table
4. Test with real games

### Performance Optimization
- Use JIT compilation for hot code paths
- Implement caching for frequently accessed memory
- Optimize instruction decoding and dispatch
- Profile memory access patterns

## Technical Details

### PowerPC Instruction Format
The emulator supports standard PowerPC instruction formats:
- I-form: Immediate instructions (ADDI, ORI, etc.)
- D-form: Memory access instructions (LWZ, STW, etc.)
- B-form: Branch instructions (BC, B, etc.)
- XL-form: Extended instructions (BCLR, etc.)

### Memory Layout
```
0x00000000 - 0x1FFFFFFF: Physical RAM (512MB)
0x80000000 - 0x9FFFFFFF: Kernel virtual space  
0x90000000 - 0xAFFFFFFF: User virtual space
0xEC800000 - 0xEC900000: GPU MMIO registers
```

### Syscall Handling
1. CPU executes `sc` instruction
2. Emulator captures syscall number from r0
3. Routes to real kernel.exe if available
4. Falls back to HLE implementation
5. Returns result in r3

## Limitations

- GPU emulation is software-based (no hardware acceleration)
- Not all PowerPC instructions implemented
- Limited game compatibility
- No audio emulation
- Requires real Xbox 360 kernel.exe for full LLE

## Contributing

1. Follow Python PEP 8 style guidelines
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Profile performance impact of changes

## Legal Notice

This emulator is for educational and research purposes. Users must legally obtain all required system files including kernel.exe. No copyrighted material is included in this repository.