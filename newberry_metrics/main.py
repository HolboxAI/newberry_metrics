from typing import Dict, Optional, Any
import json
from datetime import datetime
from dataclasses import asdict
from .bedrock_models import get_model_implementation
from .azure_models import get_azure_model_implementation
from .auth_handlers import AWSAuthHandler, AzureAuthHandler
from .prize_handlers import AWSPricingHandler, AzurePricingHandler
from .data_models import APICallMetrics, SessionMetrics
from .dashboard_manager import DashboardManager
from .session_manager import SessionManager

class TokenEstimator:
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
        
        # Initialize session manager
        identifier = region if self.provider == "aws" else connection_string
        self._session_manager = SessionManager(self.provider, self.model_id, identifier)
        self._processed_request_ids = set()  # Initialize set for tracking processed request IDs

        self._session_metrics = self._session_manager.load_session_metrics()

        if not DashboardManager.was_explicitly_stopped():
            DashboardManager.launch_dashboard()
            print(f"Dashboard should be running at: http://localhost:8501 (or will attempt to start)")

    @staticmethod
    def stop_dashboard():
        """Stop the dashboard using the DashboardManager."""
        DashboardManager.stop_dashboard()

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
        
        self._session_manager.save_session_metrics(self._session_metrics)
        
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
        self._session_manager.save_session_metrics(self._session_metrics)
