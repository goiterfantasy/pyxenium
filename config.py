# config.py

"""
Xbox 360 Emulator Configuration
Manages emulator settings and system configuration using the built-in JSON module.
"""

import os
import json
from pathlib import Path

class EmulatorConfig:
    def __init__(self, config_path="config.json"): # Changed default to .json
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self):
        """Load configuration from JSON file"""
        default_config = {
            'system': {
                'kernel_path': 'kernel.exe',
                'nand_path': 'nand.bin',
                'hdd_path': 'hdd.bin',
                'cpu_threads': 6,
                'memory_size': 512 * 1024 * 1024,
            },
            'emulation': {
                'cpu_mode': 'interpreter',
                'gpu_backend': 'software',
                'audio_backend': 'none',
                'hle_kernel': False,
                'fast_boot': False,
            },
            'debug': {
                'enable_logging': True,
                'log_level': 'INFO',
                'break_on_unknown_syscall': True,
                'trace_execution': False,
                'dump_memory': False,
            },
            'paths': {
                'game_directory': 'games/',
                'save_directory': 'saves/',
                'cache_directory': 'cache/',
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    # A proper deep merge would be better, but this is simple
                    for key, value in user_config.items():
                        if key in default_config and isinstance(default_config[key], dict):
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {self.config_path}. Using default config.")
                self._save_config(default_config)
        else:
            # Create default config file
            self._save_config(default_config)
            
        return default_config
    
    def _save_config(self, config):
        """Save configuration to JSON file"""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4) # Using indent for readability
    
    def get(self, key_path, default=None):
        """Get configuration value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
    
    def set(self, key_path, value):
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to the second-to-last key
        for key in keys[:-1]:
            config = config.setdefault(key, {})
            
        config[keys[-1]] = value
        self._save_config(self.config)

