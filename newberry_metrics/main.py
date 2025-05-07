from typing import Dict, Optional, Any, List
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
from bedrock_models import get_model_implementation
import subprocess
import webbrowser

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
    """Handles token estimation and cost calculations for different models."""
    
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

        self.launch_dashboard()

        print("Dashboard running at http://localhost:8501")

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

    def _invoke_bedrock(self, prompt: str, max_tokens: int = 500) -> Dict[str, Any]:
        """
        Invoke the Bedrock model and get the raw response from AWS.
        Automatically tracks session costs and metrics.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Dict containing the raw response from AWS and session metrics
        """
        # Get model-specific payload
        payload = self._model_implementation.get_payload(prompt, max_tokens)

        response = self._bedrock_client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload)
        )
        
        # Automatically track session costs
        session_metrics = self._track_session_cost(response)
        
        # Add session metrics to the response
        response['SessionMetrics'] = session_metrics
        
        return response

    def _process_bedrock_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the raw Bedrock response and extract relevant information.
        
        Args:
            response: The raw response from Bedrock API
            
        Returns:
            Dict containing processed response data with token counts, answer, and latency
        """
        # Use model-specific response parsing
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
        
        Args:
            response: The raw Bedrock response
            
        Returns:
            Dict containing cost information and token counts
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

    def calculate_prompt_latency(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate the latency for processing a prompt using the provided Bedrock response.
        
        Args:
            response: The raw Bedrock response
            
        Returns:
            Dict containing latency information in seconds
        """
        processed_response = self._process_bedrock_response(response)
        latency = processed_response.get("latency", 0.0)
        
        return {
            "latency_seconds": round(latency, 3),
            "latency_milliseconds": round(latency * 1000, 3),
            "timestamp": datetime.now().isoformat()
        }

    def _track_session_cost(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Track and calculate the cumulative cost and metrics for the current session.
        The session is automatically identified by the AWS credentials.
        
        Args:
            response: The raw Bedrock response
            
        Returns:
            Dict containing session metrics and current call information
        """
        cost_info = self.calculate_prompt_cost(response)
        latency = cost_info["latency"]
        
        self._session_metrics.total_cost += cost_info["cost"]
        self._session_metrics.total_latency += latency
        self._session_metrics.total_calls += 1
        
        self._session_metrics.average_cost = self._session_metrics.total_cost / self._session_metrics.total_calls
        self._session_metrics.average_latency = self._session_metrics.total_latency / self._session_metrics.total_calls
        
        api_call = APICallMetrics(
            timestamp=datetime.now().isoformat(),
            cost=round(cost_info["cost"], 6),
            latency=round(latency, 3),
            call_counter=self._session_metrics.total_calls,
            input_tokens=cost_info["input_tokens"],
            output_tokens=cost_info["output_tokens"]
        )
        self._session_metrics.api_calls.append(api_call)
        
        if self._latency_threshold_ms is not None and latency > (self._latency_threshold_ms / 1000.0):
            print(f"\n Latency Alert: Current call latency ({latency:.3f}s / {latency*1000:.0f}ms) exceeds threshold ({self._latency_threshold_ms}ms)")

        # Check total session cost against threshold
        if self._cost_threshold is not None and self._session_metrics.total_cost > self._cost_threshold:
            print(f"\n Cost Alert: Total session cost (${self._session_metrics.total_cost:.6f}) exceeds threshold (${self._cost_threshold:.6f})")
        self._save_session_metrics()
        
        return {
            "total_cost": round(self._session_metrics.total_cost, 6),
            "average_cost": round(self._session_metrics.average_cost, 6),
            "total_latency": round(self._session_metrics.total_latency, 3),
            "average_latency": round(self._session_metrics.average_latency, 3),
            "total_calls": self._session_metrics.total_calls,
            "current_call": asdict(api_call),
            "answer": cost_info["answer"]
        }

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

    def visualize_metrics(self, time_interval: str = 'hourly', save_path: Optional[str] = None) -> None:
        """
        Create bar charts for cost and latency metrics grouped by hour or day using the current session data.

        Args:
            time_interval: 'hourly' or 'daily' to specify the time grouping.
            save_path: Optional base path to save the generated plots (e.g., 'plots/session_viz').
                       Appends '_hourly.png' or '_daily.png' based on time_interval. If None, plots are shown directly.
        """
        metrics = self.get_session_metrics() 

        if not metrics.api_calls:
            print("No API call data available to visualize.")
            return

        df = pd.DataFrame([asdict(call) for call in metrics.api_calls])
        if df.empty:
            print("No API call data available to visualize.")
            return
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        if time_interval.lower() == 'hourly':
            df['time_group'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:00')
            title = "Hourly Metrics"
            color_cost = 'skyblue'
            color_latency = 'lightgreen'
        else:
            df['time_group'] = df['timestamp'].dt.strftime('%Y-%m-%d')
            title = "Daily Metrics"
            color_cost = 'tomato'
            color_latency = 'gold'
       
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

     
        cost_data = df.groupby('time_group')['cost'].sum()
        if not cost_data.empty:
            cost_data.plot(kind='bar', ax=ax1, color=color_cost)
        ax1.set_title(f'{time_interval.capitalize()} Cost Distribution')
        ax1.set_xlabel(time_interval.capitalize())
        ax1.set_ylabel('Cost ($)')
        ax1.tick_params(axis='x', rotation=45)

      
        latency_data = df.groupby('time_group')['latency'].mean()
        if not latency_data.empty:
            latency_data.plot(kind='bar', ax=ax2, color=color_latency)
        ax2.set_title(f'{time_interval.capitalize()} Average Latency Distribution')
        ax2.set_xlabel(time_interval.capitalize())
        ax2.set_ylabel('Latency (seconds)')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_path:
            save_name = f"{save_path}_{time_interval.lower()}.png"
            print(f"Saving {time_interval} plot to {save_name}")
            plt.savefig(save_name)
            plt.close(fig)
        else:
            fig.suptitle(title, fontsize=16, y=0.99)
            plt.show()

    def launch_dashboard(self):
        # Start Streamlit app in a new process
        proc = subprocess.Popen(
            ["streamlit", "run", "app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # Wait a few seconds for the server to start
        time.sleep(3)
        # Open the default Streamlit URL in the browser
        webbrowser.open("http://localhost:8501")
        return proc

if __name__ == "__main__":
    # Initialize the TokenEstimator with your model ID
    estimator = TokenEstimator(model_id="amazon.nova-pro-v1:0")

    # Example usage
    # prompt = "What is the capital of France?"
    # response = estimator._invoke_bedrock(prompt)
