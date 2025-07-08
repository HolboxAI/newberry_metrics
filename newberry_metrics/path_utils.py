"""
Path utilities for project-aware metrics storage in newberry_metrics.

This module handles determining where to store metrics data based on:
1. Environment variable override (NEWBERRY_METRICS_PATH)
2. Current working directory (default)
3. Home directory fallback
"""

import os
from pathlib import Path
from typing import Optional


def get_project_metrics_directory() -> Path:
    """
    Get the directory where metrics should be stored for the current project.
    
    Priority order:
    1. Environment variable NEWBERRY_METRICS_PATH (if set)
    2. Current working directory + .newberry_metrics
    3. Fallback to home directory + .newberry_metrics (if permission issues)
    
    Returns:
        Path: The directory where metrics should be stored
    """
    # Priority 1: Environment variable override
    if custom_path := os.getenv('NEWBERRY_METRICS_PATH'):
        try:
            metrics_dir = Path(custom_path) / ".newberry_metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            return metrics_dir
        except (PermissionError, OSError):
            # If custom path fails, continue to other options
            pass
    
    # Priority 2: Current working directory
    try:
        project_dir = Path.cwd() / ".newberry_metrics"
        project_dir.mkdir(parents=True, exist_ok=True)
        return project_dir
    except (PermissionError, OSError):
        # Priority 3: Fallback to home directory
        pass
    
    # Priority 3: Fallback to home directory
    try:
        home_dir = Path.home() / ".newberry_metrics"
        home_dir.mkdir(parents=True, exist_ok=True)
        return home_dir
    except (PermissionError, OSError):
        # If all else fails, use a temp directory
        import tempfile
        temp_dir = Path(tempfile.gettempdir()) / ".newberry_metrics"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir


def get_sessions_directory() -> Path:
    """
    Get the sessions subdirectory within the metrics directory.
    
    Returns:
        Path: The directory where session files should be stored
    """
    metrics_dir = get_project_metrics_directory()
    sessions_dir = metrics_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return sessions_dir


def get_sessions_directory_with_fallback() -> Path:
    """
    Get the sessions directory with smart fallback logic.
    
    This function checks for existing session files and falls back to
    the home directory if no files are found in the project directory.
    
    Priority:
    1. Environment variable override (if set)
    2. Current working directory (if session files exist OR no home directory files)
    3. Home directory (if it has session files and CWD doesn't)
    
    Returns:
        Path: The directory where session files should be loaded from
    """
    # Priority 1: Environment variable override
    if custom_path := os.getenv('NEWBERRY_METRICS_PATH'):
        try:
            custom_sessions_dir = Path(custom_path) / ".newberry_metrics" / "sessions"
            custom_sessions_dir.mkdir(parents=True, exist_ok=True)
            return custom_sessions_dir
        except (PermissionError, OSError):
            pass
    
    # Check current working directory
    cwd_sessions_dir = Path.cwd() / ".newberry_metrics" / "sessions"
    cwd_sessions_dir.mkdir(parents=True, exist_ok=True)
    
    # Check home directory
    home_sessions_dir = Path.home() / ".newberry_metrics" / "sessions"
    
    # Look for session files in both locations
    cwd_session_files = list(cwd_sessions_dir.glob("session_metrics_*.json")) if cwd_sessions_dir.exists() else []
    home_session_files = list(home_sessions_dir.glob("session_metrics_*.json")) if home_sessions_dir.exists() else []
    
    # Decision logic:
    if cwd_session_files:
        # CWD has session files, use it
        return cwd_sessions_dir
    elif home_session_files and not cwd_session_files:
        # Home has session files and CWD doesn't, use home
        return home_sessions_dir
    else:
        # Default to CWD (for new projects or when both are empty)
        return cwd_sessions_dir


def get_project_root() -> Path:
    """
    Get the current project root directory (where metrics are being generated).
    
    Returns:
        Path: The current working directory
    """
    return Path.cwd()


def set_custom_metrics_path(path: Optional[str] = None) -> bool:
    """
    Set a custom metrics path via environment variable.
    
    Args:
        path: Custom path for metrics storage. If None, removes the custom path.
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if path is None:
            # Remove custom path
            if 'NEWBERRY_METRICS_PATH' in os.environ:
                del os.environ['NEWBERRY_METRICS_PATH']
        else:
            # Set custom path
            os.environ['NEWBERRY_METRICS_PATH'] = str(path)
            # Test if the path is accessible
            test_dir = Path(path) / ".newberry_metrics"
            test_dir.mkdir(parents=True, exist_ok=True)
        return True
    except (PermissionError, OSError):
        return False


def get_current_metrics_info() -> dict:
    """
    Get information about current metrics storage configuration.
    
    Returns:
        dict: Information about metrics storage paths and configuration
    """
    custom_path = os.getenv('NEWBERRY_METRICS_PATH')
    metrics_dir = get_project_metrics_directory()
    sessions_dir = get_sessions_directory()
    
    return {
        'custom_path_set': custom_path is not None,
        'custom_path': custom_path,
        'metrics_directory': str(metrics_dir),
        'sessions_directory': str(sessions_dir),
        'project_root': str(get_project_root()),
        'storage_mode': _get_storage_mode()
    }


def _get_storage_mode() -> str:
    """
    Determine which storage mode is being used.
    
    Returns:
        str: Storage mode description
    """
    custom_path = os.getenv('NEWBERRY_METRICS_PATH')
    if custom_path:
        return "custom_path"
    
    try:
        project_dir = Path.cwd() / ".newberry_metrics"
        project_dir.mkdir(parents=True, exist_ok=True)
        return "project_directory"
    except (PermissionError, OSError):
        return "home_directory"
