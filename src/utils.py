"""
Utility functions for the IR Pipeline Study.
"""

import logging
from pathlib import Path
import json


def get_logger(name: str = "ir_pipeline") -> logging.Logger:
    """Get logger instance."""
    if not logging.getLogger(name).handlers:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
    return logging.getLogger(name)


def save_json(data: dict, filepath: Path) -> None:
    """Save dictionary to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: Path) -> dict:
    """Load dictionary from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)
