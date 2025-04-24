from typing import Dict, Optional, Any
from dataclasses import dataclass
import boto3
import json

@dataclass
class ModelPricing:
    """Data class to store model pricing information."""
    input_cost: float
    output_cost: float

class TokenEstimator:
    """Handles token estimation and cost calculations for different models."""
    
    def __init__(self, model_id: str, aws_access_key_id: str, aws_secret_access_key: str, region: str = "us-east-1"):
        """
        Initialize the TokenEstimator with AWS credentials and model information.
        
        Args:
            model_id: The Bedrock model ID (e.g., "amazon.nova-pro-v1:0")
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            region: AWS region (default: "us-east-1")
        """
        self.model_id = model_id
        self.region = region
        self._bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        
        # Initialize session costs dictionary
        self._session_costs: Dict[str, float] = {}

    def _invoke_bedrock(self, prompt: str) -> Dict[str, Any]:
        """
        Invoke the Bedrock model and get the response.
        
        Args:
            prompt: The input prompt
            
        Returns:
            Dict containing the response data
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

        raw_body = response["body"].read().decode("utf-8")
        result = json.loads(raw_body)

        response_data = {}
        if "output" in result and isinstance(result["output"], dict) and "message" in result["output"] and isinstance(result["output"]["message"], dict):
            if result["output"]["message"].get("role") == "assistant":
                response_data["answer"] = result["output"]["message"]["content"][0]["text"]
        else:
            response_data["answer"] = "Unexpected response format."

        if "usage" in result and isinstance(result["usage"], dict):
            response_data["inputTokens"] = result["usage"].get("inputTokens")
            response_data["outputTokens"] = result["usage"].get("outputTokens")
        else:
            response_data["inputTokens"] = None
            response_data["outputTokens"] = None

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
            response: The Bedrock response object containing token counts and answer
            
        Returns:
            Dict containing cost information and token counts
        """
        input_tokens = response.get("inputTokens", 0)
        output_tokens = response.get("outputTokens", 0)
        
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
            "answer": response.get("answer", "")
        }

    def track_session_cost(self, session_id: str, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Track and calculate the cumulative cost for a session using the provided Bedrock response.
        
        Args:
            session_id: Unique identifier for the session
            response: The Bedrock response object containing token counts and answer
            
        Returns:
            Dict containing session cost information and token counts
        """
        cost_info = self.calculate_prompt_cost(response)
        
        if session_id not in self._session_costs:
            self._session_costs[session_id] = 0.0
        self._session_costs[session_id] += cost_info["cost"]
        
        return {
            "session_cost": round(self._session_costs[session_id], 6),
            "current_cost": cost_info["cost"],
        }

    def get_session_cost(self, session_id: str) -> Optional[float]:
        """
        Get the current cost for a specific session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            Optional[float]: The session cost if it exists, None otherwise
        """
        return self._session_costs.get(session_id)


def main():
    """Example usage of the TokenEstimator class with Bedrock."""
    # These should be properly secured in production
    model_id = "amazon.nova-pro-v1:0"
    aws_access_key_id = "your_access_key_id"
    aws_secret_access_key = "your_secret_access_key"
    
    estimator = TokenEstimator(model_id, aws_access_key_id, aws_secret_access_key)
    
    # Test get_model_cost_per_million
    costs = estimator.get_model_cost_per_million()
    print("Cost per million tokens (input):", costs["input"])
    print("Cost per million tokens (output):", costs["output"])
    
    # prompt = "What is the weather in San Francisco"
    # result = estimator.calculate_prompt_cost(estimator._invoke_bedrock(prompt))
    # print("Prompt Cost:", result["cost"])

    # session_result = estimator.track_session_cost("Session1", estimator._invoke_bedrock(prompt))
    # print("\nSession cost:", session_result["session_cost"])
    # print("Current Cost:", session_result["current_cost"])


if __name__ == "__main__":
    main()
