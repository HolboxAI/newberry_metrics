try:
    from .main import TokenEstimator
except ImportError:
    from main import TokenEstimator

__all__ = ["TokenEstimator"]
