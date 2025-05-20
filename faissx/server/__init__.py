#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Unified interface for LLM providers using OpenAI format
# https://github.com/muxi-ai/faissx
#
# Copyright (C) 2025 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FAISSx Server Module

This module provides a high-performance vector database proxy server using FAISS and ZeroMQ
communication. It handles server configuration, authentication, and initialization of the
vector database service.
"""

import os
import json
from typing import Dict, Any, Optional

# Import auth module for setting API keys
from . import auth

# Default configuration values for the FAISSx server
DEFAULT_CONFIG = {
    "port": 45678,  # Default port for ZeroMQ communication
    "bind_address": "0.0.0.0",  # Listen on all network interfaces
    "data_dir": None,  # Use FAISS default storage location unless specified
    "auth_keys": {},  # Dictionary mapping API keys to tenant IDs
    "auth_file": None,  # Path to JSON file containing API key mappings
    "enable_auth": False,  # Authentication disabled by default
}

# Global configuration dictionary initialized with defaults
_config = DEFAULT_CONFIG.copy()


def configure(
    port: int = 45678,
    bind_address: str = "0.0.0.0",
    data_dir: Optional[str] = None,
    auth_keys: Optional[Dict[str, str]] = None,
    auth_file: Optional[str] = None,
    enable_auth: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Configure the FAISSx Server with custom settings.

    This function allows customization of server parameters including network settings,
    authentication, and data storage location. It handles both direct API key configuration
    and loading keys from a file.

    Args:
        port: Port to listen on (default: 45678)
        bind_address: Address to bind to (default: "0.0.0.0")
        data_dir: Directory to store FAISS indices (default: None, uses FAISS default)
        auth_keys: Dictionary mapping API keys to tenant IDs (default: {})
        auth_file: Path to JSON file containing API keys (default: None)
        enable_auth: Whether to enable authentication (default: False)
        kwargs: Additional configuration options

    Returns:
        Dict[str, Any]: Current configuration dictionary

    Raises:
        ValueError: If both auth_keys and auth_file are provided
        ValueError: If auth_file cannot be read or parsed
    """
    global _config

    # Validate that only one authentication method is specified
    if auth_keys and auth_file:
        raise ValueError("Cannot provide both auth_keys and auth_file")

    # Update configuration with provided parameters
    _config["port"] = port
    _config["bind_address"] = bind_address
    _config["data_dir"] = data_dir
    _config["auth_keys"] = auth_keys or {}
    _config["auth_file"] = auth_file
    _config["enable_auth"] = enable_auth

    # Load API keys from file if specified
    if auth_file:
        try:
            with open(auth_file, "r") as f:
                _config["auth_keys"] = json.load(f)
        except Exception as e:
            raise ValueError(
                f"Failed to load auth keys from file {auth_file}: {str(e)}"
            )

    # Initialize authentication module with API keys
    auth.set_api_keys(_config["auth_keys"])

    # Add any additional configuration options from kwargs
    for key, value in kwargs.items():
        _config[key] = value

    return _config.copy()


def get_config() -> Dict[str, Any]:
    """
    Retrieve the current server configuration.

    Returns:
        Dict[str, Any]: A copy of the current configuration dictionary
    """
    return _config.copy()


def run():
    """
    Initialize and start the FAISSx Server.

    This function:
    1. Creates necessary data directories
    2. Sets up environment variables
    3. Initializes and starts the ZeroMQ server
    4. Begins accepting client connections

    The server will run until terminated, handling vector database operations
    according to the current configuration.
    """
    # Import server module for actual server implementation
    from faissx.server.server import run_server

    # Create data directory if specified
    if _config["data_dir"]:
        os.makedirs(_config["data_dir"], exist_ok=True)

    # Configure environment variables for server process
    if _config["data_dir"]:
        os.environ["FAISSX_DATA_DIR"] = _config["data_dir"]
    os.environ["FAISSX_PORT"] = str(_config["port"])

    # Initialize and start the server with current configuration
    run_server(
        port=_config["port"],
        bind_address=_config["bind_address"],
        auth_keys=_config["auth_keys"],
        enable_auth=_config["enable_auth"],
        data_dir=_config["data_dir"],
    )
