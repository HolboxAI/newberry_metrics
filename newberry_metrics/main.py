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
from bedrock_models import get_model_implementation
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

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
        
        self._dynamodb_table_name = "BedrockSessionMetrics" # DynamoDB Table Name

        self._dynamodb = boto3.resource(
            'dynamodb',
            region_name=region,
            aws_access_key_id=frozen_credentials.access_key,
            aws_secret_access_key=frozen_credentials.secret_key,
        )

        self._session_metrics = self._load_session_metrics_from_dynamodb()

    def _hash_credentials(self, access_key: str, secret_key: str, region: str) -> str:
        """Create a hash of AWS credentials for unique session identification."""
        credential_string = f"{access_key}:{secret_key}:{region}"
        return hashlib.sha256(credential_string.encode()).hexdigest()[:8]

    def _load_session_metrics_from_dynamodb(self) -> SessionMetrics:
        """Load session metrics from DynamoDB or return default structure."""
        default_metrics = SessionMetrics(
            total_cost=0.0, average_cost=0.0, total_latency=0.0,
            average_latency=0.0, total_calls=0, api_calls=[]
        )
        try:
            table = self._dynamodb.Table(self._dynamodb_table_name)
            response = table.get_item(Key={'session_hash': self._aws_credentials_hash})

            if 'Item' in response:
                item = response['Item']
                api_calls_data = item.get("api_calls", [])
                api_calls = [APICallMetrics(
                                 timestamp=call['timestamp'],
                                 cost=float(call['cost']),
                                 latency=float(call['latency']),
                                 call_counter=int(call['call_counter']),
                                 input_tokens=int(call['input_tokens']),
                                 output_tokens=int(call['output_tokens'])
                             ) for call in api_calls_data]

                return SessionMetrics(
                    total_cost=float(item.get("total_cost", 0.0)),
                    average_cost=float(item.get("average_cost", 0.0)),
                    total_latency=float(item.get("total_latency", 0.0)),
                    average_latency=float(item.get("average_latency", 0.0)),
                    total_calls=int(item.get("total_calls", 0)),
                    api_calls=api_calls
                )
            else:
                return default_metrics
        except Exception as e:
            print(f"Warning: Could not load session metrics from DynamoDB ({self._dynamodb_table_name}): {e}")
            return default_metrics

    def _save_session_metrics_to_dynamodb(self):
        """Save session metrics to DynamoDB, converting floats to Decimals."""
        try:
            table = self._dynamodb.Table(self._dynamodb_table_name)
            metrics_dict = asdict(self._session_metrics)

            metrics_decimal_dict = self._convert_floats_to_decimals(metrics_dict)

            metrics_decimal_dict['session_hash'] = self._aws_credentials_hash

            table.put_item(Item=metrics_decimal_dict) 
        except Exception as e:
            print(f"Warning: Could not save session metrics to DynamoDB ({self._dynamodb_table_name}): {e}")

    def _convert_floats_to_decimals(self, obj):
        """Recursively walks a dict/list structure and converts float values to Decimal."""
        if isinstance(obj, dict):
            return {k: self._convert_floats_to_decimals(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_floats_to_decimals(elem) for elem in obj]
        elif isinstance(obj, float):
            return Decimal(str(obj))
        else:
            return obj

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
        self._save_session_metrics_to_dynamodb()
        
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
        self._save_session_metrics_to_dynamodb()

def visualize_metrics(metrics: SessionMetrics, time_interval: str = 'hourly', save_path: Optional[str] = None) -> None:
    """
    Create bar charts for cost and latency metrics grouped by hour or day.
    
    Args:
        metrics: SessionMetrics object containing API call data
        time_interval: 'hourly' or 'daily' to specify the time grouping
        save_path: Optional path to save the generated plots
    """
    # Convert API calls to DataFrame
    df = pd.DataFrame([asdict(call) for call in metrics.api_calls])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Group by time interval
    if time_interval == 'hourly':
        df['time_group'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:00')
        title_suffix = 'Hourly'
    else:  # daily
        df['time_group'] = df['timestamp'].dt.strftime('%Y-%m-%d')
        title_suffix = 'Daily'
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot cost metrics
    cost_data = df.groupby('time_group')['cost'].sum()
    cost_data.plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title(f'{title_suffix} Cost Distribution')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Cost ($)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot latency metrics
    latency_data = df.groupby('time_group')['latency'].mean()
    latency_data.plot(kind='bar', ax=ax2, color='lightgreen')
    ax2.set_title(f'{title_suffix} Average Latency Distribution')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Latency (seconds)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
