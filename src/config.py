"""
Simple configuration loading.
"""

import yaml
from pathlib import Path


class Config:
    def __init__(self):
        config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path, "r") as f:
            self.data = yaml.safe_load(f)
    
    def get(self, key, default=None):
        """Get config value using dot notation (e.g., 'retrieval.dense.model')."""
        keys = key.split(".")
        value = self.data
        for k in keys:
            value = value.get(k) if isinstance(value, dict) else None
            if value is None:
                return default
        return value


config = Config()
