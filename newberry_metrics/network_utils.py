"""
Network utilities for newberry_metrics.

This module contains network-related utility functions.
"""
import socket
import os


def is_port_in_use(port: int) -> bool:
    """
    Check if a port is currently in use.
    
    Args:
        port: Port number to check
        
    Returns:
        True if port is in use, False otherwise
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return False  # Port is available
        except socket.error as e:
            if e.errno == socket.errno.EADDRINUSE or (os.name == 'nt' and e.winerror == 10048):  # EADDRINUSE or WSAEADDRINUSE
                return True  # Port is in use
            # print(f"Unexpected socket error checking port {port}: {e}")  # Optional: for debugging other socket errors
            return False  # Assume available on other errors, or could raise
