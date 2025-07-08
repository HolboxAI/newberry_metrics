"""
Streamlit UI components for Newberry Metrics Dashboard.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import from newberry_metrics.ui
ui_dir = Path(__file__).parent
package_dir = ui_dir.parent
sys.path.insert(0, str(package_dir.parent))

from newberry_metrics.ui.app import main as run_dashboard

__all__ = ['run_dashboard']
