"""
FAISS Proxy Client Module

A client library for interacting with the FAISS Proxy server.
"""

from typing import Dict, Any, Optional

# Default configuration
DEFAULT_CONFIG = {
    "server": "tcp://localhost:45678",
    "api_key": None,
    "tenant_id": None,
}

# Global configuration
_config = DEFAULT_CONFIG.copy()


def configure(
    server: str = "tcp://localhost:45678",
    api_key: Optional[str] = None,
    tenant_id: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Configure the FAISS Proxy client.

    Args:
        server: ZeroMQ server URL (default: "tcp://localhost:45678")
        api_key: API key for authentication
        tenant_id: Tenant ID for multi-tenant deployments
        kwargs: Additional configuration options

    Returns:
        Current configuration
    """
    global _config

    _config["server"] = server
    _config["api_key"] = api_key
    _config["tenant_id"] = tenant_id

    # Add any additional configuration options
    for key, value in kwargs.items():
        _config[key] = value

    return _config.copy()


def get_config() -> Dict[str, Any]:
    """Get the current client configuration."""
    return _config.copy()


# Placeholder for client implementation
# The actual implementation will be developed separately
