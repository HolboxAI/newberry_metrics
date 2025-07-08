"""
Data loading functions for Newberry Metrics Dashboard.
"""

import json
from pathlib import Path
from typing import Optional


def find_latest_session_file(directory: Path, pattern: str) -> Optional[Path]:
    """Finds the most recently modified file matching the pattern in the specified directory."""
    try:
        # Ensure the directory exists before trying to glob from it
        if not directory.exists():
            # Optionally: st.info(f"Metrics directory {directory} not found. Waiting for initial metrics generation.")
            return None
        directory.mkdir(parents=True, exist_ok=True)  # Ensure it exists, useful if app starts before any metrics saved

        session_files = list(directory.glob(pattern))
        if not session_files:
            return None
        latest_file = max(session_files, key=lambda p: p.stat().st_mtime)
        return latest_file
    except Exception:
        return None


def load_session_data(file_path: Path) -> Optional[dict]:
    """Loads session data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, IOError, Exception):
        return None
