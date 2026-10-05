"""DriftSentinel - catch ML data drift before it silently breaks your models."""
from .detector import DriftDetector, DriftResult
from .monitor import StreamMonitor

__all__ = ["DriftDetector", "DriftResult", "StreamMonitor"]
__version__ = "0.1.0"
