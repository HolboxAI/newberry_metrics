"""
Dashboard management for newberry_metrics.

This module handles launching, stopping, and managing the Streamlit dashboard.
"""
import os
import subprocess
import time
import webbrowser
import signal
import errno
from pathlib import Path
from .network_utils import is_port_in_use
from .path_utils import get_project_metrics_directory


class DashboardManager:
    """Manages the Streamlit dashboard for metrics visualization."""
    
    _dashboard_browser_opened_this_session = False
    _dashboard_was_explicitly_stopped = False
    
    @classmethod
    def _get_pid_file_path(cls) -> Path:
        """Get the PID file path using project-aware directory."""
        metrics_dir = get_project_metrics_directory()
        return metrics_dir / ".newberry_dashboard.pid"
    
    @classmethod
    def launch_dashboard(cls) -> bool:
        """
        Launch the Streamlit dashboard.
        
        Returns:
            True if dashboard was launched or is already running, False otherwise
        """
        dashboard_port = 8501
        dashboard_url = f"http://localhost:{dashboard_port}"

        if is_port_in_use(dashboard_port):
            if not cls._dashboard_browser_opened_this_session:
                try:
                    webbrowser.open(dashboard_url)
                    cls._dashboard_browser_opened_this_session = True
                except webbrowser.Error:
                    pass 
            return True

        try:
            package_dir = Path(__file__).parent.resolve()
            app_py_path = package_dir / "ui" / "app.py"
            if not app_py_path.exists():
                print(f"Error: Dashboard app.py not found at {app_py_path}")
                return False

            command = ["streamlit", "run", str(app_py_path), "--server.headless", "true", "--server.port", str(dashboard_port)]
            
            creation_flags = 0
            start_new_session_flag = False
            if os.name == 'nt':  # Windows
                creation_flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            else:  # Unix-like
                start_new_session_flag = True
            
            pid_file_path = cls._get_pid_file_path()
            pid_file_path.parent.mkdir(parents=True, exist_ok=True)

            proc = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creation_flags,
                start_new_session=start_new_session_flag 
            )
            
            time.sleep(3)  # Give Streamlit a moment to start
            
            if proc.poll() is not None: 
                return False
            else:
                try:
                    with open(pid_file_path, "w") as f:
                        f.write(str(proc.pid))
                except IOError:
                    pass

                if not cls._dashboard_browser_opened_this_session:
                    try:
                        webbrowser.open(dashboard_url)
                        cls._dashboard_browser_opened_this_session = True
                    except webbrowser.Error:
                        pass
                return True

        except FileNotFoundError:
            return False
        except Exception:
            return False

    @classmethod
    def stop_dashboard(cls):
        """Stop the running dashboard."""
        pid_file_path = cls._get_pid_file_path()
        if not pid_file_path.exists():
            if is_port_in_use(8501):
                pass
            cls._dashboard_was_explicitly_stopped = True
            cls._dashboard_browser_opened_this_session = False
            return

        try:
            with open(pid_file_path, "r") as f:
                pid_str = f.read().strip()
                if not pid_str:
                    pid_file_path.unlink(missing_ok=True)
                    cls._dashboard_was_explicitly_stopped = True
                    cls._dashboard_browser_opened_this_session = False
                    return
                pid = int(pid_str)
        except (IOError, ValueError):
            pid_file_path.unlink(missing_ok=True)
            cls._dashboard_was_explicitly_stopped = True
            cls._dashboard_browser_opened_this_session = False
            return

        try:
            if os.name == 'nt':
                subprocess.run(["taskkill", "/F", "/PID", str(pid), "/T"], capture_output=True, text=True, check=False)
            else:
                os.kill(pid, signal.SIGTERM)
                time.sleep(1)
                try:
                    os.kill(pid, 0)
                    os.kill(pid, signal.SIGKILL)
                    time.sleep(0.5)
                    os.kill(pid, 0)
                except OSError as e:
                    if e.errno == errno.ESRCH:
                        pass
                    else:
                        raise
            
            pid_file_path.unlink(missing_ok=True)

        except OSError as e:
            if os.name != 'nt' and e.errno == errno.ESRCH:
                pid_file_path.unlink(missing_ok=True)
            else:
                pass
        except Exception:
            pass
        finally:
            cls._dashboard_was_explicitly_stopped = True
            cls._dashboard_browser_opened_this_session = False
    
    @classmethod
    def was_explicitly_stopped(cls) -> bool:
        """Check if dashboard was explicitly stopped."""
        return cls._dashboard_was_explicitly_stopped
