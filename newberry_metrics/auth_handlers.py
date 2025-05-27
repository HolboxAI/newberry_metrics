from abc import ABC, abstractmethod
from typing import Dict, Any
import boto3


class AuthHandlerBase(ABC):
    @abstractmethod
    def get_client(self) -> Any:
        pass

class AWSAuthHandler(AuthHandlerBase):
    def __init__(self, region: str = "us-east-1"):
        self.region = region
        
    def get_client(self):
        session = boto3.Session()
        credentials = session.get_credentials()
        if credentials is None:
            raise ValueError("No AWS credentials found")
            
        frozen_credentials = credentials.get_frozen_credentials()
        return boto3.client(
            "bedrock-runtime",
            region_name=self.region,
            aws_access_key_id=frozen_credentials.access_key,
            aws_secret_access_key=frozen_credentials.secret_key,
        )

class AzureAuthHandler(AuthHandlerBase):
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        
    def get_client(self):
        from azure.core.credentials import AzureKeyCredential
        from azure.ai.textanalytics import TextAnalyticsClient
        return TextAnalyticsClient(
            endpoint=self.connection_string,
            credential=AzureKeyCredential(self.connection_string)
        )
