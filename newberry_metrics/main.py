from typing import Dict, Optional, Any, List, Union
from dataclasses import dataclass, asdict
import boto3
import json
import os
from pathlib import Path
import hashlib
import time
from datetime import datetime
import io
from decimal import Decimal
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from .bedrock_models import get_model_implementation
from .azure_models import get_azure_model_implementation
from .auth_handlers import AWSAuthHandler, AzureAuthHandler
from .prize_handlers import AWSPricingHandler, AzurePricingHandler
import subprocess
import webbrowser
import signal
import errno

def is_port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return False # Port is available
        except socket.error as e:
            if e.errno == socket.errno.EADDRINUSE or (os.name == 'nt' and e.winerror == 10048): # EADDRINUSE or WSAEADDRINUSE
                return True # Port is in use
            # print(f"Unexpected socket error checking port {port}: {e}") # Optional: for debugging other socket errors
            return False # Assume available on other errors, or could raise

@dataclass
class APICallMetrics:
    """Data class to store metrics for a single API call."""
    timestamp: str
    cost: float
    latency: float
    call_counter: int
    input_tokens: int
    output_tokens: int

@dataclass
class SessionMetrics:
    """Data class to store overall session metrics."""
    total_cost: float
    average_cost: float
    total_latency: float
    average_latency: float
    total_calls: int
    api_calls: List[APICallMetrics]

class TokenEstimator:
    _dashboard_browser_opened_this_session = False
    _SESSIONS_BASE_DIR = Path.home() / ".newberry_metrics" / "sessions"
    PID_FILE_PATH = _SESSIONS_BASE_DIR / ".newberry_dashboard.pid"
    _dashboard_was_explicitly_stopped = False
    
    def __init__(self, 
                 model_id: str,
                 provider: str = "aws",
                 region: str = "us-east-1",
                 connection_string: Optional[str] = None,
                 cost_threshold: Optional[float] = None,
                 latency_threshold_ms: Optional[float] = None):
        """
        Initialize the TokenEstimator with model information.
        
        Args:
            model_id: The model ID (e.g., "amazon.nova-pro-v1:0" or "gpt-4")
            provider: The provider ("aws" or "azure")
            region: AWS region (default: "us-east-1")
            connection_string: Azure connection string (required for Azure)
            cost_threshold: Optional total session cost threshold for alerts
            latency_threshold_ms: Optional latency threshold in milliseconds
        """
        self.model_id = model_id
        self.provider = provider.lower()
        self.region = region
        self._cost_threshold = cost_threshold
        self._latency_threshold_ms = latency_threshold_ms
        
        # Initialize appropriate auth handler
        if self.provider == "aws":
            self._auth_handler = AWSAuthHandler(region)
        elif self.provider == "azure":
            if not connection_string:
                raise ValueError("Azure connection string is required")
            self._auth_handler = AzureAuthHandler(connection_string)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
        # Initialize appropriate model implementation
        if self.provider == "aws":
            self._model_implementation = get_model_implementation(model_id)
        else:
            self._model_implementation = get_azure_model_implementation(model_id)
            
        # Initialize appropriate pricing handler
        if self.provider == "aws":
            self._pricing_handler = AWSPricingHandler()
        else:
            self._pricing_handler = AzurePricingHandler()
        
        self._aws_credentials_hash = self._hash_credentials(
            self.provider,
            self.model_id,
            region if self.provider == "aws" else connection_string
        )
        
        TokenEstimator._SESSIONS_BASE_DIR.mkdir(parents=True, exist_ok=True)
        self._session_metrics_file = TokenEstimator._SESSIONS_BASE_DIR / f"session_metrics_{self._aws_credentials_hash}.json"
        self._processed_request_ids = set() # Initialize set for tracking processed request IDs

        self._session_metrics = self._load_session_metrics()

        if not TokenEstimator._dashboard_was_explicitly_stopped:
            TokenEstimator._launch_dashboard_static()
            print(f"Dashboard should be running at: http://localhost:8501 (or will attempt to start)")

    @staticmethod
    def _launch_dashboard_static() -> bool:
        dashboard_port = 8501
        dashboard_url = f"http://localhost:{dashboard_port}"

        if is_port_in_use(dashboard_port):
            if not TokenEstimator._dashboard_browser_opened_this_session:
                try:
                    webbrowser.open(dashboard_url)
                    TokenEstimator._dashboard_browser_opened_this_session = True
                except webbrowser.Error as e:
                    pass 
            return True

        try:
            package_dir = Path(__file__).parent.resolve()
            app_py_path = package_dir / "app.py"
            if not app_py_path.exists():
                print(f"Error: Dashboard app.py not found at {app_py_path}")
                return False

            command = ["streamlit", "run", str(app_py_path), "--server.headless", "true", "--server.port", str(dashboard_port)]
            
            creation_flags = 0
            start_new_session_flag = False
            if os.name == 'nt': # Windows
                creation_flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            else: # Unix-like
                start_new_session_flag = True
            
            TokenEstimator.PID_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

            proc = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creation_flags,
                start_new_session=start_new_session_flag 
            )
            
            time.sleep(3) # Give Streamlit a moment to start
            
            if proc.poll() is not None: 
                return False
            else:
                try:
                    with open(TokenEstimator.PID_FILE_PATH, "w") as f:
                        f.write(str(proc.pid))
                except IOError as e:
                    pass

                if not TokenEstimator._dashboard_browser_opened_this_session:
                    try:
                        webbrowser.open(dashboard_url)
                        TokenEstimator._dashboard_browser_opened_this_session = True
                    except webbrowser.Error as e:
                        pass
                return True

        except FileNotFoundError:
            return False
        except Exception as e:
            return False

    @staticmethod
    def stop_dashboard():
        if not TokenEstimator.PID_FILE_PATH.exists():
            if is_port_in_use(8501):
                 pass
            TokenEstimator._dashboard_was_explicitly_stopped = True
            TokenEstimator._dashboard_browser_opened_this_session = False
            return

        try:
            with open(TokenEstimator.PID_FILE_PATH, "r") as f:
                pid_str = f.read().strip()
                if not pid_str:
                    TokenEstimator.PID_FILE_PATH.unlink(missing_ok=True)
                    TokenEstimator._dashboard_was_explicitly_stopped = True
                    TokenEstimator._dashboard_browser_opened_this_session = False
                    return
                pid = int(pid_str)
        except (IOError, ValueError) as e:
            TokenEstimator.PID_FILE_PATH.unlink(missing_ok=True)
            TokenEstimator._dashboard_was_explicitly_stopped = True
            TokenEstimator._dashboard_browser_opened_this_session = False
            return

        try:
            if os.name == 'nt':
                result = subprocess.run(["taskkill", "/F", "/PID", str(pid), "/T"], capture_output=True, text=True, check=False)
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
            
            TokenEstimator.PID_FILE_PATH.unlink(missing_ok=True)

        except OSError as e:
            if os.name != 'nt' and e.errno == errno.ESRCH:
                TokenEstimator.PID_FILE_PATH.unlink(missing_ok=True)
            else:
                pass
        except Exception as e:
            pass
        finally:
            TokenEstimator._dashboard_was_explicitly_stopped = True
            TokenEstimator._dashboard_browser_opened_this_session = False

    def _hash_credentials(self, provider: str, model_id: str, identifier: str) -> str:
        """Create a hash of credentials for unique session identification."""
        credential_string = f"{provider}:{model_id}:{identifier}"
        return hashlib.sha256(credential_string.encode()).hexdigest()[:8]

    def _load_session_metrics(self) -> SessionMetrics:
        """Load session metrics from file or return default structure if file doesn't exist."""
        default_metrics = SessionMetrics(
            total_cost=0.0, average_cost=0.0, total_latency=0.0,
            average_latency=0.0, total_calls=0, api_calls=[]
        )
        if self._session_metrics_file.exists():
            try:
                with open(self._session_metrics_file, 'r') as f:
                    data = json.load(f)
                    api_calls_data = data.get("api_calls", [])
                    api_calls = [APICallMetrics(**call) for call in api_calls_data]
                    return SessionMetrics(
                        total_cost=data.get("total_cost", 0.0),
                        average_cost=data.get("average_cost", 0.0),
                        total_latency=data.get("total_latency", 0.0),
                        average_latency=data.get("average_latency", 0.0),
                        total_calls=data.get("total_calls", 0),
                        api_calls=api_calls
                    )
            except (json.JSONDecodeError, IOError) as e:
                return default_metrics
        return default_metrics

    def _save_session_metrics(self):
        """Save session metrics to file."""
        try:
            TokenEstimator._SESSIONS_BASE_DIR.mkdir(parents=True, exist_ok=True)
            with open(self._session_metrics_file, 'w') as f:
                metrics_dict = asdict(self._session_metrics)
                json.dump(metrics_dict, f, indent=2)
        except IOError as e:
            pass

    def _process_bedrock_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the raw Bedrock response using model-specific implementation.
        """
        return self._model_implementation.parse_response(response)

    def get_model_cost_per_million(self) -> Dict[str, float]:
        """Get the cost per million tokens for input and output."""
        return self._pricing_handler.get_model_cost_per_million(self.model_id)

    def calculate_prompt_cost(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate the cost of processing a prompt using the provided Bedrock response.
        This consumes the response body stream and uses _process_bedrock_response.
        Metrics will only be recorded once per unique Bedrock request ID.
        """
        current_timestamp = datetime.now().isoformat()
        request_id = None
        try:
            request_id = response.get('ResponseMetadata', {}).get('HTTPHeaders', {}).get('x-amzn-requestid')
        except AttributeError:
            print("This might be expected if the response stream was already consumed.")

        processed_response = self._process_bedrock_response(response)
        input_tokens = processed_response.get("inputTokens", 0)
        output_tokens = processed_response.get("outputTokens", 0)
        
        costs = self.get_model_cost_per_million()
        input_cost = (input_tokens * costs["input"]) / 1_000_000
        output_cost = (output_tokens * costs["output"]) / 1_000_000
        total_cost = input_cost + output_cost
        
        cost_and_token_info = {
            "cost": round(total_cost, 6),
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "answer": processed_response.get("answer", ""),
            "latency": processed_response.get("latency", 0.0)
        }

        should_update_metrics = True
        if request_id:
            if request_id in self._processed_request_ids:
                should_update_metrics = False

        if should_update_metrics:
            self._update_api_call_metrics(
                cost=cost_and_token_info["cost"],
                latency=cost_and_token_info["latency"],
                input_tokens=cost_and_token_info["input_tokens"],
                output_tokens=cost_and_token_info["output_tokens"],
                answer=cost_and_token_info["answer"],
                timestamp_str=current_timestamp
            )
            if request_id:
                self._processed_request_ids.add(request_id)
        
        return cost_and_token_info

    def _update_api_call_metrics(self, cost: float, latency: float, input_tokens: int, output_tokens: int, timestamp_str: str, answer: Optional[str] = None):
        """Helper method to update and save session metrics after an API call."""
        self._session_metrics.total_cost += cost
        self._session_metrics.total_latency += latency
        self._session_metrics.total_calls += 1
        
        if self._session_metrics.total_calls > 0:
            self._session_metrics.average_cost = self._session_metrics.total_cost / self._session_metrics.total_calls
            self._session_metrics.average_latency = self._session_metrics.total_latency / self._session_metrics.total_calls
        else:
            self._session_metrics.average_cost = 0.0
            self._session_metrics.average_latency = 0.0
        
        api_call_metric = APICallMetrics(
            timestamp=timestamp_str,
            cost=round(cost, 6),
            latency=round(latency, 3),
            call_counter=self._session_metrics.total_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
        self._session_metrics.api_calls.append(api_call_metric)
        
        if self._latency_threshold_ms is not None and (latency * 1000) > self._latency_threshold_ms:
            pass
        if self._cost_threshold is not None and self._session_metrics.total_cost > self._cost_threshold:
            pass
        
        self._save_session_metrics()
        
        return {
            "total_cost_session": round(self._session_metrics.total_cost, 6),
            "average_cost_session": round(self._session_metrics.average_cost, 6),
            "total_latency_session": round(self._session_metrics.total_latency, 3),
            "average_latency_session": round(self._session_metrics.average_latency, 3),
            "total_calls_session": self._session_metrics.total_calls,
            "current_call_metrics": asdict(api_call_metric),
            "answer": answer
        }

    def get_response(self, prompt: str, max_tokens: int = 500) -> Dict[str, Any]:
        """
        Invokes the configured model with a prompt, tracks metrics,
        and returns a dictionary containing the model's answer and metrics.
        """
        payload_body_dict = self._model_implementation.get_payload(prompt, max_tokens)
        
        if self.provider == "aws":
            raw_response = self._auth_handler.get_client().invoke_model(
                modelId=self.model_id,
                contentType="application/json", 
                accept="*/*", 
                body=json.dumps(payload_body_dict)
            )
        else:  # Azure
            raw_response = self._auth_handler.get_client().analyze_text(
                documents=[{"id": "1", "text": prompt}],
                **payload_body_dict
            )
        
        return raw_response

    def get_session_metrics(self) -> SessionMetrics:
        """
        Get all metrics for the current session.
        The session is automatically identified by the AWS credentials.
        
        Returns:
            SessionMetrics object containing all session metrics
        """
        return self._session_metrics

    def reset_session_metrics(self) -> None:
        """
        Reset all metrics for the current session.
        The session is automatically identified by the AWS credentials.
        """
        self._session_metrics = SessionMetrics(
            total_cost=0.0,
            average_cost=0.0,
            total_latency=0.0,
            average_latency=0.0,
            total_calls=0,
            api_calls=[]
        )
        self._save_session_metrics()
