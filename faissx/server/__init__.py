"""
FAISSx Server Module

This module provides a high-performance vector database proxy server
using FAISS and ZeroMQ communication.
"""

import os
import json
from typing import Dict, Any, Optional

# Import auth module for setting API keys
from . import auth

# Default configuration
DEFAULT_CONFIG = {
    "port": 45678,
    "bind_address": "0.0.0.0",
    "data_dir": None,  # Use FAISS default unless specified
    "auth_keys": {},  # API key to tenant mapping
    "auth_file": None,  # Path to JSON file with API keys
    "enable_auth": False,
}

# Global configuration
_config = DEFAULT_CONFIG.copy()


def configure(
    port: int = 45678,
    bind_address: str = "0.0.0.0",
    data_dir: Optional[str] = None,
    auth_keys: Optional[Dict[str, str]] = None,
    auth_file: Optional[str] = None,
    enable_auth: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Configure the FAISSx Server.

    Args:
        port: Port to listen on (default: 45678)
        bind_address: Address to bind to (default: "0.0.0.0")
        data_dir: Directory to store FAISS indices (default: None, uses FAISS default)
        auth_keys: Dictionary mapping API keys to tenant IDs (default: {})
        auth_file: Path to JSON file containing API keys (default: None)
        enable_auth: Whether to enable authentication (default: False)
        kwargs: Additional configuration options

    Returns:
        Current configuration

    Raises:
        ValueError: If both auth_keys and auth_file are provided
    """
    global _config

    # Check that only one auth method is provided
    if auth_keys and auth_file:
        raise ValueError("Cannot provide both auth_keys and auth_file")

    _config["port"] = port
    _config["bind_address"] = bind_address
    _config["data_dir"] = data_dir
    _config["auth_keys"] = auth_keys or {}
    _config["auth_file"] = auth_file
    _config["enable_auth"] = enable_auth

    # Load API keys from file if provided
    if auth_file:
        try:
            with open(auth_file, 'r') as f:
                _config["auth_keys"] = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load auth keys from file {auth_file}: {str(e)}")

    # Set API keys in the auth module
    auth.set_api_keys(_config["auth_keys"])

    # Add any additional configuration options
    for key, value in kwargs.items():
        _config[key] = value

    return _config.copy()


def get_config() -> Dict[str, Any]:
    """Get the current server configuration."""
    return _config.copy()


def run():
    """
    Run the FAISSx Server with the current configuration.

    This will start the ZeroMQ server and begin accepting connections.
    """
    # Use the server module directly
    from .server import run_server

    # Ensure data directory exists if specified
    if _config["data_dir"]:
        os.makedirs(_config["data_dir"], exist_ok=True)

    # Set environment variables from config
    if _config["data_dir"]:
        os.environ["FAISS_DATA_DIR"] = _config["data_dir"]
    os.environ["faissx_PORT"] = str(_config["port"])

    # Start the server
    run_server(
        port=_config["port"],
        bind_address=_config["bind_address"],
        auth_keys=_config["auth_keys"],
        enable_auth=_config["enable_auth"],
        data_dir=_config["data_dir"],
    )
