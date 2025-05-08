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
import subprocess
import webbrowser
import atexit

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
    _dashboard_process = None
    _dashboard_launched = False
    _atexit_registered = False
    
    def __init__(self, model_id: str, region: str = "us-east-1",
                 cost_threshold: Optional[float] = None,
                 latency_threshold_ms: Optional[float] = None):
        """
        Initialize the TokenEstimator with model information.
        AWS credentials will be loaded from the system configuration.
        
        Args:
            model_id: The Bedrock model ID (e.g., "amazon.nova-pro-v1:0")
            region: AWS region (default: "us-east-1")
            cost_threshold: Optional total session cost threshold for alerts.
            latency_threshold_ms: Optional latency threshold in milliseconds for individual call alerts.
        """
        self.model_id = model_id
        self.region = region
        self._cost_threshold = cost_threshold
        self._latency_threshold_ms = latency_threshold_ms
        
        session = boto3.Session()
        credentials = session.get_credentials()
        if credentials is None:
            raise ValueError("No AWS credentials found. Please configure AWS credentials.")
            
        frozen_credentials = credentials.get_frozen_credentials()
        
        self._bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            aws_access_key_id=frozen_credentials.access_key,
            aws_secret_access_key=frozen_credentials.secret_key,
        )
        
        self._model_implementation = get_model_implementation(model_id)
        
        self._aws_credentials_hash = self._hash_credentials(
            frozen_credentials.access_key,
            frozen_credentials.secret_key,
            region
        )
        
        self._session_metrics_file = Path(f"session_metrics_{self._aws_credentials_hash}.json")

        self._session_metrics = self._load_session_metrics()

        if not TokenEstimator._dashboard_launched:
            TokenEstimator._dashboard_process = TokenEstimator._launch_dashboard_static()
            if TokenEstimator._dashboard_process:
                TokenEstimator._dashboard_launched = True
                print(f"Newberry Metrics Dashboard available at http://localhost:8501 (PID: {TokenEstimator._dashboard_process.pid})")
                if not TokenEstimator._atexit_registered:
                    atexit.register(TokenEstimator._shutdown_dashboard_static)
                    TokenEstimator._atexit_registered = True
            else:
                print("Warning: Failed to start Newberry Metrics dashboard.")

    @staticmethod
    def _launch_dashboard_static():
        try:
            script_dir = Path(__file__).parent
            app_py_path = script_dir / "app.py"
            if not app_py_path.exists():
                print(f"Error: Dashboard application file (app.py) not found at {app_py_path}")
                return None

            proc = subprocess.Popen(
                ["streamlit", "run", str(app_py_path), "--server.headless", "true", "--server.port", "8501"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            time.sleep(3) 
            
            if proc.poll() is None:
                webbrowser.open("http://localhost:8501")
                return proc
            else:
                stderr_output = proc.stderr.read().decode() if proc.stderr else "No stderr output"
                print(f"Error starting Streamlit dashboard: {stderr_output}")
                return None
        except FileNotFoundError:
            print("Error: 'streamlit' command not found. Ensure Streamlit is installed and in your PATH.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred while launching dashboard: {e}")
            return None

    @staticmethod
    def _shutdown_dashboard_static():
        if TokenEstimator._dashboard_process and TokenEstimator._dashboard_process.poll() is None:
            print("Newberry Metrics: Shutting down dashboard server...")
            TokenEstimator._dashboard_process.terminate()
            try:
                TokenEstimator._dashboard_process.wait(timeout=5)
                print("Newberry Metrics: Dashboard server shut down gracefully.")
            except subprocess.TimeoutExpired:
                print("Newberry Metrics: Dashboard server did not respond to terminate, forcing kill...")
                TokenEstimator._dashboard_process.kill()
                TokenEstimator._dashboard_process.wait()
                print("Newberry Metrics: Dashboard server killed.")
            TokenEstimator._dashboard_process = None
            TokenEstimator._dashboard_launched = False # Reset for potential re-runs
        elif TokenEstimator._dashboard_process:
            print("Newberry Metrics: Dashboard server was already stopped.")

    def _hash_credentials(self, access_key: str, secret_key: str, region: str) -> str:
        """Create a hash of AWS credentials for unique session identification."""
        credential_string = f"{access_key}:{secret_key}:{region}"
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
                print(f"Warning: Could not load session metrics from {self._session_metrics_file}: {e}")
                return default_metrics
        return default_metrics

    def _save_session_metrics(self):
        """Save session metrics to file."""
        try:
            with open(self._session_metrics_file, 'w') as f:
                metrics_dict = asdict(self._session_metrics)
                json.dump(metrics_dict, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save session metrics to {self._session_metrics_file}: {e}")

    def _process_bedrock_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the raw Bedrock response using model-specific implementation.
        """
        return self._model_implementation.parse_response(response)

    def get_model_cost_per_million(self) -> Dict[str, float]:
        """
        Get the cost per million tokens for input and output for the current model in us-east-1 region.
        
        Returns:
            Dict containing input and output costs per million tokens
        """
        model_pricing = {
            "amazon.nova-pro-v1:0": {"input": 0.003, "output": 0.012},  # $0.003/$0.012 per 1K tokens
            "amazon.nova-micro-v1:0": {"input": 0.000035, "output": 0.00014},  # $0.000035/$0.00014 per 1K tokens
            "anthropic.claude-3-sonnet-20240229-v1:0": {"input": 0.003, "output": 0.015},  # $0.003/$0.015 per 1K tokens
            "anthropic.claude-3-haiku-20240307-v1:0": {"input": 0.00025, "output": 0.00125},  # $0.00025/$0.00125 per 1K tokens
            "anthropic.claude-3-opus-20240229-v1:0": {"input": 0.015, "output": 0.075},  # $0.015/$0.075 per 1K tokens
            "meta.llama2-13b-chat-v1": {"input": 0.00075, "output": 0.001},  # $0.00075/$0.001 per 1K tokens
            "meta.llama2-70b-chat-v1": {"input": 0.00195, "output": 0.00256},  # $0.00195/$0.00256 per 1K tokens
            "ai21.jamba-1-5-large-v1:0": {"input": 0.0125, "output": 0.0125},  # $0.0125 per 1K tokens
            "cohere.command-r-v1:0": {"input": 0.0005, "output": 0.0015},  # $0.0005/$0.0015 per 1K tokens
            "cohere.command-r-plus-v1:0": {"input": 0.003, "output": 0.015},  # $0.003/$0.015 per 1K tokens
            "mistral.mistral-7b-instruct-v0:2": {"input": 0.0002, "output": 0.0006},  # $0.0002/$0.0006 per 1K tokens
            "mistral.mixtral-8x7b-instruct-v0:1": {"input": 0.0007, "output": 0.0021},  # $0.0007/$0.0021 per 1K tokens
        }
        
        if self.model_id not in model_pricing:
            raise ValueError(f"Pricing not available for model: {self.model_id}. Please add pricing information in get_model_cost_per_million.")
            
        # Convert from per 1K tokens to per 1M tokens
        return {
            "input": model_pricing[self.model_id]["input"] * 1000,
            "output": model_pricing[self.model_id]["output"] * 1000
        }

    def calculate_prompt_cost(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate the cost of processing a prompt using the provided Bedrock response.
        This consumes the response body stream and uses _process_bedrock_response.
        """
        processed_response = self._process_bedrock_response(response)
        input_tokens = processed_response.get("inputTokens", 0)
        output_tokens = processed_response.get("outputTokens", 0)
        
        costs = self.get_model_cost_per_million()
        input_cost = (input_tokens * costs["input"]) / 1_000_000
        output_cost = (output_tokens * costs["output"]) / 1_000_000
        total_cost = input_cost + output_cost
        
        return {
            "cost": round(total_cost, 6),
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "answer": processed_response.get("answer", ""),
            "latency": processed_response.get("latency", 0.0)
        }

    def _update_api_call_metrics(self, cost: float, latency: float, input_tokens: int, output_tokens: int, answer: Optional[str] = None):
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
            timestamp=datetime.now().isoformat(),
            cost=round(cost, 6),
            latency=round(latency, 3),
            call_counter=self._session_metrics.total_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
        self._session_metrics.api_calls.append(api_call_metric)
        
        if self._latency_threshold_ms is not None and (latency * 1000) > self._latency_threshold_ms:
            print(f"\nLatency Alert: Current call latency ({latency:.3f}s / {latency*1000:.0f}ms) exceeds threshold ({self._latency_threshold_ms}ms)")
        if self._cost_threshold is not None and self._session_metrics.total_cost > self._cost_threshold:
            print(f"\nCost Alert: Total session cost (${self._session_metrics.total_cost:.6f}) exceeds threshold (${self._cost_threshold:.6f})")
        
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
        Invokes the configured Bedrock model with a prompt, tracks metrics,
        and returns a dictionary containing the model's answer and metrics for the call.
        This method uses the model implementation from bedrock_models.py for payload and parsing.
        """

        # 1. Get payload using model_implementation from bedrock_models.py
        payload_body_dict = self._model_implementation.get_payload(prompt, max_tokens)

        # 2. Call Bedrock
        raw_bedrock_response_obj = self._bedrock_client.invoke_model(
            modelId=self.model_id,
            contentType="application/json", 
            accept="*/*", 
            body=json.dumps(payload_body_dict) # Ensure payload is a JSON string
        )

        # 3. Calculate cost and extract tokens (consumes response body)
        # calculate_prompt_cost uses _process_bedrock_response which uses _model_implementation.parse_response
        cost_and_token_info = self.calculate_prompt_cost(raw_bedrock_response_obj)
        # cost_and_token_info contains: cost, input_tokens, output_tokens, answer, bedrock_latency (from model parsing)

        # 4. Update overall session metrics and get summary for this call
        call_summary_and_metrics = self._update_api_call_metrics(
            cost=cost_and_token_info["cost"],
            latency=cost_and_token_info["latency"],
            input_tokens=cost_and_token_info["input_tokens"],
            output_tokens=cost_and_token_info["output_tokens"],
            answer=cost_and_token_info["answer"]
        )
        
        return call_summary_and_metrics

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

if __name__ == "__main__":
    print("Initializing Newberry Metrics TokenEstimator...")
    print("The dashboard will be launched. The script will continue running.")
    print("Press Ctrl+C in this terminal to stop the script and the dashboard.")
    
    estimator = None # Define estimator outside try block for access in finally
    try:
        estimator = TokenEstimator(model_id="anthropic.claude-3-haiku-20240307-v1:0", region="us-east-1")
        
        print("\nTokenEstimator is active. Dashboard is running.")
        print("Make further calls to the 'estimator' object to see live updates in the dashboard (after refresh).")
        print("Example: In a Python interpreter, you could do 'estimator.get_response(\"Another prompt\")'")
        print("This script will now wait. Press Ctrl+C to exit.")
        
        while True:
            # Keep the main script alive.
            # Check if the dashboard process (if we were directly managing it here) is still alive.
            # However, with static management, we don't need to poll TokenEstimator._dashboard_process here.
            # atexit will handle its shutdown.
            time.sleep(5) # Sleep for a bit to reduce CPU usage.
            # You could add logic here to periodically make more calls or check for other tasks.
            # For now, it just keeps the script alive.

    except KeyboardInterrupt:
        print("\nCtrl+C received. Shutting down application...")
    except Exception as e:
        print(f"An critical error occurred in the main application: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nNewberry Metrics application is exiting.")
