"""File loading helpers for diagnosis scripts."""

import json
from typing import Dict

__all__ = ["load_json_file"]


def load_json_file(file_path: str) -> Dict:
    """Load a JSON file.

    Args:
        file_path: JSON file path

    Returns:
        Parsed dict
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
