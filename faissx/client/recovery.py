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
FAISSx client error recovery and reconnection module.

This module provides mechanisms for handling connection failures and automatically
reconnecting to the FAISSx server after network disruptions. It implements a robust
reconnection strategy with exponential backoff and jitter to handle temporary
network issues and server unavailability.
"""

import time
import logging
import threading
import random
from typing import Optional, Callable, Dict, Any

from .client import get_client

logger = logging.getLogger(__name__)

# Global recovery settings - controls retry behavior and reconnection strategy
_recovery_settings = {
    "max_retries": 5,  # Maximum number of retry attempts before giving up
    "initial_backoff": 1.0,  # Initial backoff time in seconds before first retry
    "max_backoff": 60.0,  # Maximum backoff time in seconds to cap exponential growth
    "backoff_factor": 2.0,  # Multiplier for backoff on each retry (exponential growth)
    "jitter": 0.2,  # Random jitter factor to avoid thundering herd problem
    "enabled": True,  # Master switch for recovery functionality
    "reconnect_timeout": 300.0,  # Maximum time to keep trying reconnects (5 minutes)
    "auto_reconnect": True,  # Whether to automatically start reconnection on failure
}

# Tracks the current state of reconnection attempts
_reconnection_status = {
    "attempting": False,  # Flag indicating if reconnection is in progress
    "last_attempt": 0.0,  # Timestamp of the most recent reconnection attempt
    "attempts": 0,  # Counter for number of reconnection attempts made
    "thread": None,  # Reference to the background reconnection thread
}

# Maintains the current connection state and configuration
_connection_state = {
    "last_connection_time": 0.0,  # Timestamp of last successful connection
    "is_connected": False,  # Current connection status
    "server_url": None,  # Target server URL for reconnection
    "api_key": None,  # Authentication key for server
    "tenant_id": None,  # Tenant identifier for multi-tenant setups
}

# Callback registries for connection state changes
_on_reconnect_callbacks = []  # Functions to execute after successful reconnection
_on_disconnect_callbacks = []  # Functions to execute after disconnection

# Thread synchronization lock for safe concurrent access
_lock = threading.RLock()


def configure_recovery(
    max_retries: Optional[int] = None,
    initial_backoff: Optional[float] = None,
    max_backoff: Optional[float] = None,
    backoff_factor: Optional[float] = None,
    jitter: Optional[float] = None,
    enabled: Optional[bool] = None,
    reconnect_timeout: Optional[float] = None,
    auto_reconnect: Optional[bool] = None,
) -> None:
    """
    Configure the recovery and reconnection behavior.

    This function allows fine-tuning of the reconnection strategy by adjusting
    parameters like retry counts, backoff timing, and automatic reconnection
    behavior. All parameters are optional - only specified values will be updated.

    Args:
        max_retries: Maximum number of retry attempts before giving up
        initial_backoff: Initial backoff time in seconds before first retry
        max_backoff: Maximum backoff time in seconds to cap exponential growth
        backoff_factor: Multiplier for backoff on each retry (exponential growth)
        jitter: Random jitter factor to avoid thundering herd problem
        enabled: Master switch for recovery functionality
        reconnect_timeout: Maximum time to keep trying reconnects (seconds)
        auto_reconnect: Whether to automatically start reconnection on failure
    """
    with _lock:
        # Update only the specified settings
        if max_retries is not None:
            _recovery_settings["max_retries"] = max_retries
        if initial_backoff is not None:
            _recovery_settings["initial_backoff"] = initial_backoff
        if max_backoff is not None:
            _recovery_settings["max_backoff"] = max_backoff
        if backoff_factor is not None:
            _recovery_settings["backoff_factor"] = backoff_factor
        if jitter is not None:
            _recovery_settings["jitter"] = jitter
        if enabled is not None:
            _recovery_settings["enabled"] = enabled
        if reconnect_timeout is not None:
            _recovery_settings["reconnect_timeout"] = reconnect_timeout
        if auto_reconnect is not None:
            _recovery_settings["auto_reconnect"] = auto_reconnect


def get_recovery_settings() -> Dict[str, Any]:
    """
    Get the current recovery settings.

    Returns a copy of the current recovery configuration to prevent external
    modification of internal state.

    Returns:
        Dict[str, Any]: A copy of the current recovery settings
    """
    with _lock:
        return dict(_recovery_settings)


def on_reconnect(callback: Callable[[], None]) -> None:
    """
    Register a callback to be called when reconnection succeeds.

    The callback will be executed after a successful reconnection is established.
    Multiple callbacks can be registered and will be executed in registration order.

    Args:
        callback: Function to call after successful reconnection
    """
    with _lock:
        _on_reconnect_callbacks.append(callback)


def on_disconnect(callback: Callable[[], None]) -> None:
    """
    Register a callback to be called when a disconnection is detected.

    The callback will be executed when the connection is lost or fails.
    Multiple callbacks can be registered and will be executed in registration order.

    Args:
        callback: Function to call after disconnection
    """
    with _lock:
        _on_disconnect_callbacks.append(callback)


def is_connected() -> bool:
    """
    Check if the client is currently connected.

    Returns:
        bool: True if connected, False otherwise
    """
    with _lock:
        return _connection_state["is_connected"]


def set_connected(connected: bool = True) -> None:
    """
    Update the connection state and trigger appropriate callbacks.

    This function manages the connection state and executes registered callbacks
    when the connection state changes. It also handles automatic reconnection
    if enabled.

    Args:
        connected: Whether the client is connected
    """
    with _lock:
        prev_state = _connection_state["is_connected"]
        _connection_state["is_connected"] = connected

        if connected:
            # Update connection timestamp and reset reconnection state
            _connection_state["last_connection_time"] = time.time()
            _reconnection_status["attempting"] = False
            _reconnection_status["attempts"] = 0

            # Execute reconnect callbacks if state changed from disconnected
            if not prev_state:
                for callback in _on_reconnect_callbacks:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"Error in reconnect callback: {e}")

        # Handle disconnection
        elif prev_state and not connected:
            # Execute disconnect callbacks
            for callback in _on_disconnect_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error in disconnect callback: {e}")

            # Start automatic reconnection if enabled
            if (
                _recovery_settings["auto_reconnect"]
                and not _reconnection_status["attempting"]
            ):
                start_reconnection()


def calculate_backoff(attempt: int) -> float:
    """
    Calculate the backoff time for a given attempt using exponential backoff with jitter.

    The backoff time increases exponentially with each attempt, but is capped at
    max_backoff. Random jitter is added to prevent synchronized retry attempts.

    Args:
        attempt: The current attempt number (0-based)

    Returns:
        float: Time to wait in seconds before next attempt
    """
    # Calculate exponential backoff with maximum cap
    backoff = min(
        _recovery_settings["max_backoff"],
        _recovery_settings["initial_backoff"]
        * (_recovery_settings["backoff_factor"] ** attempt),
    )

    # Add random jitter to prevent thundering herd
    jitter_amount = backoff * _recovery_settings["jitter"]
    backoff += random.uniform(-jitter_amount, jitter_amount)

    return max(0.1, backoff)  # Ensure minimum 100ms backoff


def with_retry(func: Callable, *args, **kwargs) -> Any:
    """
    Execute a function with automatic retry on failure.

    This decorator-like function implements the retry logic with exponential backoff.
    It will retry the operation up to max_retries times if it fails due to
    connection errors.

    Args:
        func: Function to execute
        *args: Arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Any: Result of the function if successful

    Raises:
        Exception: The last exception if all retries fail
    """
    if not _recovery_settings["enabled"]:
        # Skip retry logic if recovery is disabled
        return func(*args, **kwargs)

    last_exception = None

    # Try up to max_retries times
    for attempt in range(_recovery_settings["max_retries"] + 1):
        try:
            # Attempt the operation
            result = func(*args, **kwargs)

            # If we got here, it succeeded - update connection state
            set_connected(True)

            return result

        except Exception as e:
            last_exception = e

            # Update connection state
            set_connected(False)

            # If this was the last attempt, re-raise the exception
            if attempt >= _recovery_settings["max_retries"]:
                raise

            # Calculate backoff time
            backoff = calculate_backoff(attempt)

            # Log the failure and retry attempt
            logger.warning(
                f"Operation failed (attempt {attempt+1}/{_recovery_settings['max_retries']+1}): "
                f"{e}. Retrying in {backoff:.2f} seconds..."
            )

            # Wait before retrying
            time.sleep(backoff)

    # We should never reach here due to the raise in the loop,
    # but just in case...
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry failed for unknown reason")


def start_reconnection() -> None:
    """
    Start a background thread to attempt reconnection to the server.
    """
    with _lock:
        # Don't start if already attempting
        if _reconnection_status["attempting"]:
            return

        # Don't start if no server URL is set (not configured)
        if not _connection_state["server_url"]:
            return

        # Mark as attempting reconnection
        _reconnection_status["attempting"] = True
        _reconnection_status["attempts"] = 0
        _reconnection_status["last_attempt"] = time.time()

        # Start background thread for reconnection
        thread = threading.Thread(
            target=_reconnection_worker, name="FAISSx-Reconnect", daemon=True
        )
        _reconnection_status["thread"] = thread
        thread.start()


def _reconnection_worker() -> None:
    """
    Background worker thread function for reconnection attempts.
    """
    start_time = time.time()

    # Keep trying until timeout or success
    while (
        _reconnection_status["attempting"]
        and time.time() - start_time < _recovery_settings["reconnect_timeout"]
    ):
        try:
            # Get connection parameters
            with _lock:
                server_url = _connection_state["server_url"]
                api_key = _connection_state["api_key"]
                tenant_id = _connection_state["tenant_id"]
                attempt = _reconnection_status["attempts"]

                # Update attempt count
                _reconnection_status["attempts"] += 1
                _reconnection_status["last_attempt"] = time.time()

            # Calculate backoff time
            backoff = calculate_backoff(attempt)

            # Log reconnection attempt
            logger.info(
                f"Attempting reconnection (attempt {attempt+1}). "
                f"Server: {server_url}"
            )

            # Get the client
            client = get_client()

            # Try to reconnect - this will create a new client if needed
            if client is None or not client.ping():
                from . import client as client_module

                client_module.configure(
                    server=server_url, api_key=api_key, tenant_id=tenant_id
                )
                client = get_client()

                # Verify connection with ping
                if client and client.ping():
                    logger.info(f"Reconnection successful after {attempt+1} attempts")
                    set_connected(True)
                    return

            # Wait before next attempt
            logger.info(
                f"Reconnection attempt failed. Retrying in {backoff:.2f} seconds..."
            )
            time.sleep(backoff)

        except Exception as e:
            logger.error(f"Error during reconnection attempt: {e}")
            time.sleep(1.0)  # Brief pause after error

    # If we get here, reconnection failed or timed out
    with _lock:
        _reconnection_status["attempting"] = False
        logger.warning(
            f"Reconnection attempts exhausted after "
            f"{_reconnection_status['attempts']} tries. Giving up."
        )


# Helper functions to wrap common operations with retry
def retry_operation(func_name: str, *args, **kwargs) -> Any:
    """
    Retry a client operation with the configured retry policy.

    Args:
        func_name: Name of the client method to call
        *args: Arguments to pass to the method
        **kwargs: Keyword arguments to pass to the method

    Returns:
        Result of the operation

    Raises:
        RuntimeError: If the client is not available
        Exception: For other errors after retries are exhausted
    """
    client = get_client()
    if client is None:
        raise RuntimeError("FAISSx client not available - please configure first")

    # Get the method from the client
    func = getattr(client, func_name)

    # Execute with retry
    return with_retry(func, *args, **kwargs)


# ----- Public API for recovery status and control -----


def is_recovering() -> bool:
    """Check if reconnection is currently being attempted."""
    with _lock:
        return _reconnection_status["attempting"]


def last_attempt_time() -> float:
    """Get the timestamp of the last reconnection attempt."""
    with _lock:
        return _reconnection_status["last_attempt"]


def attempts_count() -> int:
    """Get the number of reconnection attempts made."""
    with _lock:
        return _reconnection_status["attempts"]


def cancel_recovery() -> None:
    """Cancel any ongoing reconnection attempts."""
    with _lock:
        _reconnection_status["attempting"] = False


def force_reconnect() -> None:
    """Force an immediate reconnection attempt."""
    set_connected(False)
    start_reconnection()


# Register with client configuration
def _store_connection_params(
    server: str, api_key: Optional[str], tenant_id: Optional[str]
) -> None:
    """
    Store connection parameters for reconnection.

    This should be called by the client module when configure() is called.

    Args:
        server: Server URL
        api_key: API key (optional)
        tenant_id: Tenant ID (optional)
    """
    with _lock:
        _connection_state["server_url"] = server
        _connection_state["api_key"] = api_key
        _connection_state["tenant_id"] = tenant_id
        _connection_state["is_connected"] = True
        _connection_state["last_connection_time"] = time.time()


# Test the connection periodically
def _start_connection_monitor() -> None:
    """
    Start a background thread to monitor the connection status.
    """

    def monitor_loop():
        while True:
            try:
                # Sleep first to allow initial connection
                time.sleep(30.0)  # Check every 30 seconds

                # Skip if recovery is disabled
                if not _recovery_settings["enabled"]:
                    continue

                # Skip if not configured
                if not _connection_state["server_url"]:
                    continue

                # Skip if reconnection is already in progress
                if _reconnection_status["attempting"]:
                    continue

                # Check connection
                client = get_client()
                if client:
                    try:
                        # Ping to verify connection
                        client.ping()
                        set_connected(True)
                    except Exception:
                        # Ping failed, mark as disconnected
                        set_connected(False)
            except Exception as e:
                logger.error(f"Error in connection monitor: {e}")

    # Start the monitor thread
    thread = threading.Thread(target=monitor_loop, name="FAISSx-Monitor", daemon=True)
    thread.start()


# Initialize connection monitoring when module is imported
_start_connection_monitor()
