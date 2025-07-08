"""
Session management for newberry_metrics.

This module handles session persistence, loading, and saving metrics data.
"""
import json
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import asdict
from .data_models import SessionMetrics, APICallMetrics
from .path_utils import get_sessions_directory


class SessionManager:
    """Manages session metrics persistence and loading."""
    
    def __init__(self, provider: str, model_id: str, identifier: str):
        """
        Initialize session manager.
        
        Args:
            provider: Provider name (aws, azure, etc.)
            model_id: Model identifier
            identifier: Unique identifier (region, connection string, etc.)
        """
        self._credentials_hash = self._hash_credentials(provider, model_id, identifier)
        self._sessions_dir = get_sessions_directory()
        self._session_metrics_file = self._sessions_dir / f"session_metrics_{self._credentials_hash}.json"
    
    def _hash_credentials(self, provider: str, model_id: str, identifier: str) -> str:
        """Create a hash of credentials for unique session identification."""
        credential_string = f"{provider}:{model_id}:{identifier}"
        return hashlib.sha256(credential_string.encode()).hexdigest()[:8]
    
    def load_session_metrics(self) -> SessionMetrics:
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
            except (json.JSONDecodeError, IOError):
                return default_metrics
        return default_metrics
    
    def save_session_metrics(self, session_metrics: SessionMetrics):
        """Save session metrics to file."""
        try:
            self._sessions_dir.mkdir(parents=True, exist_ok=True)
            with open(self._session_metrics_file, 'w') as f:
                metrics_dict = asdict(session_metrics)
                json.dump(metrics_dict, f, indent=2)
        except IOError:
            pass
