from abc import ABC, abstractmethod
from typing import Dict, Any

class AzureModelBase(ABC):
    @abstractmethod
    def get_payload(self, prompt: str, max_tokens: int = 500) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        pass

class AzureOpenAIModel(AzureModelBase):
    def get_payload(self, prompt: str, max_tokens: int = 500) -> Dict[str, Any]:
        return {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
    
    def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "answer": response.choices[0].message.content,
            "inputTokens": response.usage.prompt_tokens,
            "outputTokens": response.usage.completion_tokens,
            "latency": response.usage.total_tokens / 1000.0  # Approximate latency as it's not present in the response object as bedrock
        }


def get_azure_model_implementation(model_id: str) -> AzureModelBase:
    """Get the appropriate Azure model implementation based on model ID."""
    # For now, default to AzureOpenAIModel for all Azure models
    # This can be expanded to support different Azure model types
    return AzureOpenAIModel()
