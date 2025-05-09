from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import json

class BedrockModelBase(ABC):
    """Abstract base class for Bedrock model implementations."""
    
    @abstractmethod
    def get_payload(self, prompt: str, max_tokens: int = 500) -> Dict[str, Any]:
        """
        Get the model-specific payload format.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Dict containing the formatted payload for the specific model
        """
        pass
    
    @abstractmethod
    def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the model-specific response format.
        
        Args:
            response: Raw response from Bedrock API
            
        Returns:
            Dict containing standardized response with:
            - answer: str
            - inputTokens: int
            - outputTokens: int
            - latency: float
        """
        pass

class Claude3Model(BedrockModelBase):
    """Implementation for Anthropic Claude 3 models (Opus, Sonnet, Haiku)."""
    
    def get_payload(self, prompt: str, max_tokens: int = 500) -> Dict[str, Any]:
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
    
    def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        raw_body = response["body"].read().decode("utf-8")
        parsed_response = json.loads(raw_body)
        
        # Extract answer
        answer = ""
        if "content" in parsed_response:
            answer = parsed_response.get("content", "")
        elif "messages" in parsed_response and len(parsed_response["messages"]) > 0:
            last_message = parsed_response["messages"][-1]
            if last_message.get("role") == "assistant":
                answer = last_message.get("content", "")
        
        # Extract token counts
        input_tokens = parsed_response.get("usage", {}).get("input_tokens", 0)
        output_tokens = parsed_response.get("usage", {}).get("output_tokens", 0)
        
        # Extract latency
        latency = float(response['ResponseMetadata']['HTTPHeaders'].get('x-amzn-bedrock-invocation-latency', 0)) / 1000.0
        
        return {
            "answer": answer,
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "latency": latency
        }

class Claude2Model(BedrockModelBase):
    """Implementation for Anthropic Claude 2 models."""
    
    def get_payload(self, prompt: str, max_tokens: int = 500) -> Dict[str, Any]:
        return {
            "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
            "max_tokens_to_sample": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9
        }
    
    def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        raw_body = response["body"].read().decode("utf-8")
        parsed_response = json.loads(raw_body)
        
        # Extract answer
        answer = parsed_response.get("completion", "")
        
        # Extract token counts (Claude 2 doesn't provide token counts directly)
        # Estimate based on characters
        input_tokens = len(parsed_response.get("prompt", "")) // 4 or 0
        output_tokens = len(answer) // 4 or 0
        
        # Extract latency
        latency = float(response['ResponseMetadata']['HTTPHeaders'].get('x-amzn-bedrock-invocation-latency', 0)) / 1000.0
        
        return {
            "answer": answer,
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "latency": latency
        }

class AI21Model(BedrockModelBase):
    """Implementation for AI21 models (Jamba)."""
    
    def get_payload(self, prompt: str, max_tokens: int = 500) -> Dict[str, Any]:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.8
        }
    
    def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        raw_body = response["body"].read().decode("utf-8")
        parsed_response = json.loads(raw_body)
        
        # Extract answer from the response
        answer = ""
        if "messages" in parsed_response and len(parsed_response["messages"]) > 0:
            last_message = parsed_response["messages"][-1]
            if last_message.get("role") == "assistant":
                answer = last_message.get("content", "")
        
        # Extract token counts
        usage = parsed_response.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        
        # Extract latency
        latency = float(response['ResponseMetadata']['HTTPHeaders'].get('x-amzn-bedrock-invocation-latency', 0)) / 1000.0
        
        return {
            "answer": answer,
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "latency": latency
        }

class CohereModel(BedrockModelBase):
    """Implementation for Cohere Command models."""
    
    def get_payload(self, prompt: str, max_tokens: int = 500) -> Dict[str, Any]:
        return {
            "message": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "p": 0.8,  # equivalent to top_p
            "return_prompt": False,
            "stream": False
        }
    
    def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        raw_body = response["body"].read().decode("utf-8")
        parsed_response = json.loads(raw_body)
        
        # Extract answer
        answer = parsed_response.get("text", "")
        
        # Extract token counts
        input_tokens = parsed_response.get("meta", {}).get("prompt_tokens", 0)
        output_tokens = parsed_response.get("meta", {}).get("response_tokens", 0)
        
        # Extract latency
        latency = float(response['ResponseMetadata']['HTTPHeaders'].get('x-amzn-bedrock-invocation-latency', 0)) / 1000.0
        
        return {
            "answer": answer,
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "latency": latency
        }

class AmazonTitanModel(BedrockModelBase):
    """Implementation for Amazon Titan models."""
    
    def get_payload(self, prompt: str, max_tokens: int = 500) -> Dict[str, Any]:
        return {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "temperature": 0.7,
                "topP": 0.9,
                "stopSequences": []
            }
        }
    
    def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        raw_body = response["body"].read().decode("utf-8")
        parsed_response = json.loads(raw_body)
        
        # Extract answer
        answer = parsed_response.get("results", [{}])[0].get("outputText", "")
        
        # Extract token counts
        input_tokens = parsed_response.get("inputTextTokenCount", 0)
        output_tokens = len(answer.split()) or 0  # Rough estimate if not provided
        
        # Extract latency
        latency = float(response['ResponseMetadata']['HTTPHeaders'].get('x-amzn-bedrock-invocation-latency', 0)) / 1000.0
        
        return {
            "answer": answer,
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "latency": latency
        }

class AmazonNovaModel(BedrockModelBase):
    """Implementation for Amazon Nova models (Nova Pro, Nova Micro)."""
    
    def get_payload(self, prompt: str, max_tokens: int = 500) -> Dict[str, Any]:
        return {
            "inferenceConfig": {
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9
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
    
    def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        raw_body = response["body"].read().decode("utf-8")
        parsed_response = json.loads(raw_body)
        
        # Extract answer
        answer = ""
        if "output" in parsed_response and "message" in parsed_response["output"]:
            message = parsed_response["output"]["message"]
            if message.get("role") == "assistant" and "content" in message:
                for content in message["content"]:
                    if content.get("type") == "text" or "text" in content:
                        answer += content.get("text", "")
        
        # Extract token counts
        input_tokens = parsed_response.get("usage", {}).get("inputTokens", 0)
        output_tokens = parsed_response.get("usage", {}).get("outputTokens", 0)
        
        # Extract latency
        latency = float(response['ResponseMetadata']['HTTPHeaders'].get('x-amzn-bedrock-invocation-latency', 0)) / 1000.0
        
        return {
            "answer": answer,
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "latency": latency
        }

class MetaLlamaModel(BedrockModelBase):
    """Implementation for Meta Llama models."""
    
    def get_payload(self, prompt: str, max_tokens: int = 500) -> Dict[str, Any]:
        return {
            "prompt": prompt,
            "max_gen_len": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9
        }
    
    def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        raw_body = response["body"].read().decode("utf-8")
        parsed_response = json.loads(raw_body)
        
        # Extract answer
        answer = parsed_response.get("generation", "")
        
        # Extract token counts (not directly provided by Llama)
        # Estimate based on input/output text length
        input_tokens = len(parsed_response.get("prompt", "").split()) or 0
        output_tokens = len(answer.split()) or 0
        
        # Extract latency
        latency = float(response['ResponseMetadata']['HTTPHeaders'].get('x-amzn-bedrock-invocation-latency', 0)) / 1000.0
        
        return {
            "answer": answer,
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "latency": latency
        }

class MistralModel(BedrockModelBase):
    """Implementation for Mistral AI models."""
    
    def get_payload(self, prompt: str, max_tokens: int = 500) -> Dict[str, Any]:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9
        }
    
    def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        raw_body = response["body"].read().decode("utf-8")
        parsed_response = json.loads(raw_body)
        
        # Extract answer
        answer = ""
        if "messages" in parsed_response and len(parsed_response["messages"]) > 0:
            last_message = parsed_response["messages"][-1]
            if last_message.get("role") == "assistant":
                answer = last_message.get("content", "")
        
        # Extract token counts
        input_tokens = parsed_response.get("usage", {}).get("input_tokens", 0)
        output_tokens = parsed_response.get("usage", {}).get("output_tokens", 0)
        
        # Extract latency
        latency = float(response['ResponseMetadata']['HTTPHeaders'].get('x-amzn-bedrock-invocation-latency', 0)) / 1000.0
        
        return {
            "answer": answer,
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "latency": latency
        }

class DeepSeekModel(BedrockModelBase):
    """Implementation for DeepSeek models."""
    
    def get_payload(self, prompt: str, max_tokens: int = 500) -> Dict[str, Any]:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9
        }
    
    def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        raw_body = response["body"].read().decode("utf-8")
        parsed_response = json.loads(raw_body)
        
        # Extract answer
        answer = ""
        if "choices" in parsed_response and len(parsed_response["choices"]) > 0:
            choice = parsed_response["choices"][0]
            if "message" in choice and choice["message"].get("role") == "assistant":
                answer = choice["message"].get("content", "")
        
        # Extract token counts
        input_tokens = parsed_response.get("usage", {}).get("prompt_tokens", 0)
        output_tokens = parsed_response.get("usage", {}).get("completion_tokens", 0)
        
        # Extract latency
        latency = float(response['ResponseMetadata']['HTTPHeaders'].get('x-amzn-bedrock-invocation-latency', 0)) / 1000.0
        
        return {
            "answer": answer,
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "latency": latency
        }

def get_model_implementation(model_id: str) -> BedrockModelBase:
    """
    Get the appropriate model implementation based on the model ID.
    
    Args:
        model_id: The Bedrock model ID
        
    Returns:
        BedrockModelBase implementation for the specified model
    """
    model_id_lower = model_id.lower()
    
    if model_id_lower.startswith("anthropic.claude-3"):
        return Claude3Model()
    elif model_id_lower.startswith("anthropic.claude-2"):
        return Claude2Model()
    elif model_id_lower.startswith("ai21"):
        return AI21Model()
    elif model_id_lower.startswith("cohere"):
        return CohereModel()
    elif model_id_lower.startswith("amazon.titan"):
        return AmazonTitanModel()
    elif model_id_lower.startswith("amazon.nova"):
        return AmazonNovaModel()
    elif model_id_lower.startswith("meta.llama"):
        return MetaLlamaModel()
    elif model_id_lower.startswith("mistral"):
        return MistralModel()
    elif model_id_lower.startswith("deepseek"):
        return DeepSeekModel()
    else:
        raise ValueError(f"Unsupported model ID: {model_id}. Please add a custom implementation for this model.") 
        