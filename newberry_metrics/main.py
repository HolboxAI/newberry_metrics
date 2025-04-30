from typing import Dict, Optional, Any, List
from dataclasses import dataclass, asdict
import boto3
import json
import os
from pathlib import Path
import hashlib
import time
from datetime import datetime

@dataclass
class ModelPricing:
    """Data class to store model pricing information."""
    input_cost: float
    output_cost: float

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
    
    def __init__(self, model_id: str, region: str = "us-east-1"):
        """
        Initialize the TokenEstimator with model information.
        AWS credentials will be loaded from the system configuration.
        
        Args:
            model_id: The Bedrock model ID (e.g., "amazon.nova-pro-v1:0")
            region: AWS region (default: "us-east-1")
        """
        self.model_id = model_id
        self.region = region
        
        # Initialize AWS client using default credential chain
        self._bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=region
        )
        
        # Get AWS credentials from the session
        session = boto3.Session()
        credentials = session.get_credentials()
        if credentials is None:
            raise ValueError("No AWS credentials found. Please configure AWS credentials.")
            
        # Create a unique identifier for the AWS credentials
        self._aws_credentials_hash = self._hash_credentials(
            credentials.access_key,
            credentials.secret_key,
            region
        )
        
        # Initialize session metrics from file or create new dictionary
        self._session_metrics_file = Path(f"session_metrics_{self._aws_credentials_hash}.json")
        self._session_metrics = self._load_session_metrics()

    def _hash_credentials(self, access_key: str, secret_key: str, region: str) -> str:
        """Create a hash of AWS credentials for unique session identification."""
        credential_string = f"{access_key}:{secret_key}:{region}"
        return hashlib.sha256(credential_string.encode()).hexdigest()[:8]

    def _load_session_metrics(self) -> SessionMetrics:
        """Load session metrics from file or return default structure if file doesn't exist."""
        default_metrics = SessionMetrics(
            total_cost=0.0,
            average_cost=0.0,
            total_latency=0.0,
            average_latency=0.0,
            total_calls=0,
            api_calls=[]
        )
        
        if self._session_metrics_file.exists():
            try:
                with open(self._session_metrics_file, 'r') as f:
                    data = json.load(f)
                    # Convert API calls back to APICallMetrics objects
                    api_calls = [APICallMetrics(**call) for call in data["api_calls"]]
                    return SessionMetrics(
                        total_cost=data["total_cost"],
                        average_cost=data["average_cost"],
                        total_latency=data["total_latency"],
                        average_latency=data["average_latency"],
                        total_calls=data["total_calls"],
                        api_calls=api_calls
                    )
            except json.JSONDecodeError:
                return default_metrics
        return default_metrics

    def _save_session_metrics(self):
        """Save session metrics to file."""
        with open(self._session_metrics_file, 'w') as f:
            # Convert SessionMetrics to dict, including nested APICallMetrics
            metrics_dict = asdict(self._session_metrics)
            json.dump(metrics_dict, f, indent=2)

    def _invoke_bedrock(self, prompt: str) -> Dict[str, Any]:
        """
        Invoke the Bedrock model and get the raw response from AWS.
        Automatically tracks session costs and metrics.
        
        Args:
            prompt: The input prompt
            
        Returns:
            Dict containing the raw response from AWS and session metrics
        """
        payload = {
            "inferenceConfig": {
                "max_new_tokens": 500
            },
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }

        response = self._bedrock_client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload)
        )
        
        # Automatically track session costs
        session_metrics = self.track_session_cost(response)
        
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
        response_data = {}
        
        # Read and parse the response body
        raw_body = response["body"].read().decode("utf-8")
        parsed_response = json.loads(raw_body)
        
        # Extract answer from response
        if "output" in parsed_response and isinstance(parsed_response["output"], dict) and "message" in parsed_response["output"] and isinstance(parsed_response["output"]["message"], dict):
            if parsed_response["output"]["message"].get("role") == "assistant":
                response_data["answer"] = parsed_response["output"]["message"]["content"][0]["text"]
        else:
            response_data["answer"] = "Unexpected response format."

        # Extract token counts
        if "usage" in parsed_response and isinstance(parsed_response["usage"], dict):
            response_data["inputTokens"] = parsed_response["usage"].get("inputTokens")
            response_data["outputTokens"] = parsed_response["usage"].get("outputTokens")
        else:
            response_data["inputTokens"] = None
            response_data["outputTokens"] = None

        # Extract latency from response headers
        response_data["latency"] = float(response['ResponseMetadata']['HTTPHeaders']['x-amzn-bedrock-invocation-latency']) / 1000.0

        return response_data

    def get_model_cost_per_million(self) -> Dict[str, float]:
        """
        Get the cost per million tokens for input and output for the current model in us-east-1 region.
        
        Returns:
            Dict containing input and output costs per million tokens
        """
        # Bedrock model pricing for us-east-1 region (as of latest pricing)
        # Format: {"model_id": {"input": cost_per_1K_input_tokens, "output": cost_per_1K_output_tokens}}
        model_pricing = {
            "amazon.nova-pro-v1:0": {"input": 0.003, "output": 0.012},  # $0.003/$0.012 per 1K tokens
            "amazon.nova-micro-v1:0": {"input": 0.000035, "output": 0.00014},  # $0.000035/$0.00014 per 1K tokens
            "anthropic.claude-3-sonnet-20240229-v1:0": {"input": 0.003, "output": 0.015},  # $0.003/$0.015 per 1K tokens
            "anthropic.claude-3-haiku-20240307-v1:0": {"input": 0.00025, "output": 0.00125},  # $0.00025/$0.00125 per 1K tokens
            "meta.llama2-13b-chat-v1": {"input": 0.00075, "output": 0.001},  # $0.00075/$0.001 per 1K tokens
            "meta.llama2-70b-chat-v1": {"input": 0.00195, "output": 0.00256},  # $0.00195/$0.00256 per 1K tokens
        }
        
        if self.model_id not in model_pricing:
            raise ValueError(f"Pricing not available for model: {self.model_id}")
            
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

    def track_session_cost(self, response: Dict[str, Any]) -> Dict[str, Any]:
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
        
        # Update total metrics
        self._session_metrics.total_cost += cost_info["cost"]
        self._session_metrics.total_latency += latency
        self._session_metrics.total_calls += 1
        
        # Calculate averages
        self._session_metrics.average_cost = self._session_metrics.total_cost / self._session_metrics.total_calls
        self._session_metrics.average_latency = self._session_metrics.total_latency / self._session_metrics.total_calls
        
        # Create new API call metrics
        api_call = APICallMetrics(
            timestamp=datetime.now().isoformat(),
            cost=round(cost_info["cost"], 6),
            latency=round(latency, 3),
            call_counter=self._session_metrics.total_calls,
            input_tokens=cost_info["input_tokens"],
            output_tokens=cost_info["output_tokens"]
        )
        self._session_metrics.api_calls.append(api_call)
        
        # Save the updated metrics
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
