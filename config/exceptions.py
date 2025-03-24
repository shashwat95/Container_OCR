"""Custom exceptions for the OCR system."""

class ConfigurationError(Exception):
    """Raised when there is an error in the configuration."""
    pass

class DatabaseError(Exception):
    """Raised when there is a database-related error."""
    pass

class CameraError(Exception):
    """Raised when there is a camera-related error."""
    pass

class MLModelError(Exception):
    """Raised when there is an error with ML models."""
    pass

class ValidationError(Exception):
    """Raised when validation fails."""
    pass

class StorageError(Exception):
    """Raised when there is an error with file storage."""
    pass

class ProcessingError(Exception):
    """Raised when there is an error during processing."""
    pass 