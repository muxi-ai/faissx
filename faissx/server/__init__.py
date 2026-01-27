#!/usr/bin/env python3
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

The module follows a configuration-then-run pattern:
1. Call configure() to set up server parameters (optional, uses defaults otherwise)
2. Call run() to start the server with configured settings

Key Features:
- Multi-tenant authentication with API keys
- Persistent storage for vector indices (optional)
- ZeroMQ-based binary protocol for high performance
- Comprehensive input validation and error handling
- Support for both inline auth keys and JSON auth files

Example Usage:
    # Basic server with defaults
    import faissx.server as server
    server.run()

    # Custom configuration
    server.configure(
        port=8080,
        enable_auth=True,
        auth_keys={"key1": "tenant1", "key2": "tenant2"}
    )
    server.run()

    # Load auth from file
    server.configure(
        enable_auth=True,
        auth_file="/path/to/auth.json"
    )
    server.run()
"""

import json
from typing import Any, Dict, Optional

# Import auth module for setting API keys
from . import auth

# Configuration constants - these provide sensible defaults for most use cases
DEFAULT_PORT = (
    45678  # ZeroMQ default port, chosen to avoid conflicts with common services
)
DEFAULT_BIND_ADDRESS = (
    "0.0.0.0"  # Bind to all network interfaces for maximum accessibility
)
DEFAULT_LOG_LEVEL = "WARNING"  # Balance between useful info and noise reduction

# Valid logging levels supported by Python's logging module
# Using a set for O(1) lookup performance during validation
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

# Default configuration values for the FAISSx server
# This dictionary serves as the template for all server configurations
DEFAULT_CONFIG: Dict[str, Any] = {
    "port": DEFAULT_PORT,  # TCP port for ZeroMQ socket binding
    "bind_address": DEFAULT_BIND_ADDRESS,  # Network interface address (0.0.0.0 = all)
    "data_dir": None,  # Persistent storage directory (None = in-memory only)
    "auth_keys": {},  # Direct API key mapping: {"api_key": "tenant_id", ...}
    "auth_file": None,  # Alternative: load auth keys from JSON file
    "enable_auth": False,  # Security flag: requires API keys when True
    "log_level": DEFAULT_LOG_LEVEL,  # Controls verbosity of server logging
}

# Global configuration state - modified by configure() and read by run()
# Using copy() to ensure DEFAULT_CONFIG remains immutable
_config = DEFAULT_CONFIG.copy()


def configure(
    port: int = DEFAULT_PORT,
    bind_address: str = DEFAULT_BIND_ADDRESS,
    data_dir: Optional[str] = None,
    auth_keys: Optional[Dict[str, str]] = None,
    auth_file: Optional[str] = None,
    enable_auth: bool = False,
    log_level: str = DEFAULT_LOG_LEVEL,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Configure the FAISSx Server with custom settings.

    This function modifies the global server configuration that will be used when run()
    is called. It validates all inputs, handles authentication setup, and manages the
    relationship between direct auth keys and auth file loading.

    The function is designed to be called once before run(), but can be called multiple
    times to reconfigure the server (though this will replace the entire configuration).

    Args:
        port: TCP port for ZeroMQ socket (1-65535, default: 45678)
        bind_address: Network interface to bind to (default: "0.0.0.0" for all interfaces)
        data_dir: Directory for persistent FAISS index storage (None = in-memory only)
        auth_keys: Direct API key to tenant mapping {"key": "tenant"} (default: {})
        auth_file: Path to JSON file with auth keys (mutually exclusive with auth_keys)
        enable_auth: Whether to require API key authentication (default: False)
        log_level: Logging verbosity (DEBUG/INFO/WARNING/ERROR/CRITICAL, default: WARNING)
        kwargs: Additional configuration options for future extensibility

    Returns:
        Dict[str, Any]: Complete current configuration (safe copy, modifications won't
                        affect server)

    Raises:
        ValueError: If both auth_keys and auth_file are provided (ambiguous configuration)
        ValueError: If port is outside valid range (1-65535)
        ValueError: If bind_address is empty or whitespace-only
        ValueError: If log_level is not a recognized logging level
        ValueError: If auth_file contains invalid JSON or wrong data structure
        FileNotFoundError: If auth_file path does not exist
        PermissionError: If auth_file exists but cannot be read due to permissions

    Note:
        The function modifies global state (_config). In multi-threaded environments,
        ensure configure() is called before starting any threads that might call run().
    """
    global _config

    # Step 1: Validate all inputs before making any changes to global state
    # This ensures we fail fast if there are any configuration errors
    _validate_configuration_inputs(port, bind_address, auth_keys, auth_file, log_level)

    # Step 2: Update core configuration parameters
    # These are the fundamental server settings that control network and storage behavior
    _config["port"] = port
    _config["bind_address"] = bind_address
    _config["data_dir"] = data_dir  # None means in-memory indices only

    # Step 3: Handle authentication configuration
    # Initialize with provided auth_keys or empty dict if None
    _config["auth_keys"] = auth_keys or {}
    _config["auth_file"] = auth_file  # Store file path for reference
    _config["enable_auth"] = enable_auth

    # Step 4: Normalize log level to uppercase for consistency
    # Handle both string and potential future non-string log level types
    _config["log_level"] = (
        log_level.upper() if isinstance(log_level, str) else log_level
    )

    # Step 5: Override auth_keys with file contents if auth_file is specified
    # This happens after basic config update so file loading errors don't leave config in bad state
    if auth_file:
        _config["auth_keys"] = _load_auth_keys_from_file(auth_file)

    # Step 6: Initialize the authentication subsystem with the final auth keys
    # This makes the auth keys available to the auth module for request validation
    auth.set_api_keys(_config["auth_keys"])

    # Step 7: Apply any additional configuration options from kwargs
    # This allows for future extensibility without changing the function signature
    for key, value in kwargs.items():
        _config[key] = value

    # Step 8: Return a safe copy of the configuration
    # Callers can inspect the config without accidentally modifying server state
    return _config.copy()


def _validate_configuration_inputs(
    port: int,
    bind_address: str,
    auth_keys: Optional[Dict[str, str]],
    auth_file: Optional[str],
    log_level: str,
) -> None:
    """
    Validate configuration inputs for the server.

    Performs comprehensive validation of all configuration parameters to ensure they meet
    the requirements for a functional server. This function is designed to fail fast
    with descriptive error messages rather than allowing invalid configurations to
    propagate and cause runtime errors later.

    Args:
        port: Network port number (must be in range 1-65535)
        bind_address: Network address to bind to (must be non-empty string)
        auth_keys: Dictionary of API keys to tenant IDs (can be None or empty)
        auth_file: Path to authentication file (mutually exclusive with auth_keys)
        log_level: Logging level string (must be valid Python logging level)

    Raises:
        ValueError: If any validation check fails, with specific error message describing
                   the problem and expected valid values

    Note:
        This function does not validate that auth_file exists or is readable - that
        validation happens in _load_auth_keys_from_file() to provide more specific
        error types (FileNotFoundError, PermissionError).
    """
    # Authentication method validation: prevent ambiguous configuration
    # Both auth_keys and auth_file would create confusion about which takes precedence
    if auth_keys and auth_file:
        raise ValueError("Cannot provide both auth_keys and auth_file")

    # Port validation: ensure we can bind to the specified port
    # Valid TCP port range is 1-65535 (port 0 is reserved)
    if not (1 <= port <= 65535):
        raise ValueError(f"Port must be between 1 and 65535, got: {port}")

    # Bind address validation: ensure we have a valid network interface specification
    # Empty or whitespace-only addresses would cause binding failures
    if not bind_address or not bind_address.strip():
        raise ValueError("Bind address cannot be empty")

    # Log level validation: ensure compatibility with Python's logging module
    # Convert to uppercase for case-insensitive comparison
    if log_level.upper() not in VALID_LOG_LEVELS:
        raise ValueError(
            f"Invalid log level '{log_level}'. Must be one of: {', '.join(VALID_LOG_LEVELS)}"
        )


def _load_auth_keys_from_file(auth_file: str) -> Dict[str, str]:
    """
    Load authentication keys from a JSON file.

    Loads and validates a JSON file containing API key to tenant ID mappings. The file
    must contain a JSON object where both keys and values are strings. This function
    provides detailed error handling for common file access and format issues.

    Expected JSON format:
        {
            "api_key_1": "tenant_id_1",
            "api_key_2": "tenant_id_2",
            ...
        }

    Args:
        auth_file: Path to the JSON file containing auth keys (relative or absolute path)

    Returns:
        Dict[str, str]: Dictionary mapping API keys to tenant IDs, ready for use in
                       authentication

    Raises:
        FileNotFoundError: If the specified file path does not exist
        PermissionError: If the file exists but cannot be read due to filesystem permissions
        ValueError: If the file contains invalid JSON, or if the JSON structure is not
                   a flat dictionary of string key-value pairs

    Note:
        The function validates not just JSON syntax but also the expected data structure.
        Non-string keys or values will result in ValueError for clear error reporting.
    """
    try:
        # Open file with explicit UTF-8 encoding for consistent behavior across platforms
        # Using context manager ensures file is properly closed even if errors occur
        with open(auth_file, encoding="utf-8") as f:
            auth_data = json.load(f)
    except FileNotFoundError:
        # Re-raise with more descriptive message for better debugging
        raise FileNotFoundError(f"Authentication file not found: {auth_file}")
    except PermissionError:
        # Re-raise with context about what operation failed
        raise PermissionError(
            f"Permission denied reading authentication file: {auth_file}"
        )
    except json.JSONDecodeError as e:
        # JSON syntax errors get their own specific error type for clarity
        raise ValueError(f"Invalid JSON in authentication file {auth_file}: {str(e)}")
    except Exception as e:
        # Catch-all for other file system or unexpected errors
        raise ValueError(f"Failed to load auth keys from file {auth_file}: {str(e)}")

    # Data structure validation: ensure we got a dictionary (JSON object)
    # Arrays, strings, numbers, etc. are invalid for our use case
    if not isinstance(auth_data, dict):
        raise ValueError(
            f"Authentication file must contain a JSON object, got: {type(auth_data).__name__}"
        )

    # Content validation: ensure all keys and values are strings
    # This prevents issues with non-string keys that can't be used for authentication
    for key, value in auth_data.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(
                f"Authentication file must contain string key-value pairs, "
                f"got key type: {type(key).__name__}, value type: {type(value).__name__}"
            )

    # Return validated data ready for use in authentication
    return auth_data


def get_config() -> Dict[str, Any]:
    """
    Retrieve the current server configuration.

    Returns a safe copy of the current server configuration that can be inspected
    without risk of accidentally modifying the server's internal state. Useful for
    debugging, logging, or displaying current settings.

    Returns:
        Dict[str, Any]: A deep copy of the current configuration dictionary containing
                       all settings that will be used when run() is called

    Note:
        The returned dictionary is a copy - modifications to it will not affect the
        server configuration. Use configure() to change server settings.

    Example:
        config = get_config()
        print(f"Server will listen on port {config['port']}")
        print(f"Authentication enabled: {config['enable_auth']}")
    """
    # Return a copy to prevent accidental modification of server state
    return _config.copy()


def run():
    """
    Run the FAISSx server with the current configuration.

    This function starts the main server process using the configuration established
    by configure(). If configure() was not called previously, default settings will
    be used. The server will run indefinitely until interrupted (Ctrl+C) or killed.

    The function blocks and does not return until the server is shut down. For
    production deployments, consider running this in a proper process manager like
    systemd, supervisord, or Docker.

    Raises:
        Various exceptions depending on configuration errors, network issues, or
        runtime problems. Common issues include:
        - Port already in use (address binding errors)
        - Permission denied (trying to bind to privileged ports < 1024)
        - Invalid data directory path or permissions
        - Missing dependencies or import errors

    Note:
        This function imports the actual server implementation module only when
        called to avoid circular import issues and to keep the module initialization
        lightweight.

    Example:
        # Start with defaults
        faissx.server.run()

        # Start with custom config
        faissx.server.configure(port=8080, enable_auth=True)
        faissx.server.run()
    """
    # Extract configuration values for clarity and to avoid repeated dict lookups
    # This also makes the function call signature more explicit
    port = _config["port"]
    bind_address = _config["bind_address"]
    data_dir = _config["data_dir"]
    auth_keys = _config["auth_keys"]
    enable_auth = _config["enable_auth"]
    log_level = _config["log_level"]

    # Deferred import to avoid circular dependencies between modules
    # The server module might import from this module, so we delay the import
    from faissx.server.server import run_server

    # Start the server with all configured parameters
    # This call blocks until the server is shut down
    run_server(
        port=port,
        bind_address=bind_address,
        auth_keys=auth_keys,
        enable_auth=enable_auth,
        data_dir=data_dir,
        log_level=log_level,
    )
