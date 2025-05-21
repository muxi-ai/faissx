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

Architecture:
-------------
The recovery system consists of three main components:

1. Retry Mechanism: Functions wrapped with 'with_retry()' automatically retry on failure
   using exponential backoff with jitter to prevent thundering herd problem.

2. Connection Monitoring: A background thread periodically checks connection health
   by sending ping requests and triggers reconnection when disconnections are detected.

3. Reconnection Worker: A separate thread handles reconnection attempts with backoff,
   allowing the main application to continue functioning during recovery.

The module uses thread-safe state management with immutable configuration objects and
proper locking to ensure consistency across threads.

Usage:
------
# Configure recovery settings
configure_recovery(max_retries=3, initial_backoff=2.0)

# Register callbacks for connection events
on_reconnect(lambda: print("Reconnected!"))
on_disconnect(lambda: print("Disconnected!"))

# Retry operations automatically
result = retry_operation("search", query_vector, k=10)

# Manual recovery management (if needed)
if not is_connected():
    force_reconnect()

# Check recovery status
if is_recovering():
    print(f"Reconnection in progress. Attempts: {attempts_count()}")
"""

import time
import logging
import threading
import random
from typing import Optional, Callable, Dict, Any, NamedTuple, List, TypeVar, cast

from .client import get_client

logger = logging.getLogger(__name__)

# Type variable for generic function return
T = TypeVar('T')


class RecoverySettings(NamedTuple):
    """
    Immutable configuration for recovery behavior.

    Attributes:
        max_retries: Maximum number of retry attempts
        initial_backoff: Initial delay between retries in seconds
        max_backoff: Maximum delay between retries in seconds
        backoff_factor: Multiplier for exponential backoff
        jitter: Random jitter factor (0-1) to prevent thundering herd
        enabled: Whether recovery is enabled
        reconnect_timeout: Maximum time to attempt reconnection
        auto_reconnect: Whether to automatically attempt reconnection
    """

    max_retries: int = 5
    initial_backoff: float = 1.0
    max_backoff: float = 60.0
    backoff_factor: float = 2.0
    jitter: float = 0.2
    enabled: bool = True
    reconnect_timeout: float = 300.0
    auto_reconnect: bool = True


class ReconnectionStatus(NamedTuple):
    """
    Current state of reconnection attempts.

    Attributes:
        attempting: Whether a reconnection attempt is in progress
        last_attempt: Timestamp of last reconnection attempt
        attempts: Number of reconnection attempts made
        thread: Background thread handling reconnection
    """

    attempting: bool = False
    last_attempt: float = 0.0
    attempts: int = 0
    thread: Optional[threading.Thread] = None


class ConnectionState(NamedTuple):
    """
    Current connection state and configuration.

    Attributes:
        last_connection_time: Timestamp of last successful connection
        is_connected: Whether currently connected
        server_url: Server URL to connect to
        api_key: API key for authentication
        tenant_id: Tenant ID for multi-tenancy
    """

    last_connection_time: float = 0.0
    is_connected: bool = False
    server_url: Optional[str] = None
    api_key: Optional[str] = None
    tenant_id: Optional[str] = None


# Initialize state with default values
_recovery_settings = RecoverySettings()._asdict()
_reconnection_status = ReconnectionStatus()._asdict()
_connection_state = ConnectionState()._asdict()

# Callback registries for connection state changes
_on_reconnect_callbacks: List[Callable[[], None]] = []
_on_disconnect_callbacks: List[Callable[[], None]] = []

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

    This function allows you to customize how the FAISSx client handles network disruptions
    and server unavailability. Any parameters not explicitly specified will retain their
    current values.

    The exponential backoff strategy follows this formula:
    delay = min(max_backoff, initial_backoff * (backoff_factor ^ attempt_number))

    A random jitter is then added to prevent synchronized retry storms:
    final_delay = delay ± (delay * jitter * random_factor)

    Args:
        max_retries: Maximum number of retry attempts before giving up
        initial_backoff: Initial delay between retries in seconds
        max_backoff: Maximum delay between retries in seconds
        backoff_factor: Multiplier for exponential backoff (base of exponentiation)
        jitter: Random jitter factor (0-1) to prevent synchronized retries
        enabled: Whether recovery is enabled globally
        reconnect_timeout: Maximum time in seconds to attempt reconnection
        auto_reconnect: Whether to automatically attempt reconnection on failure

    Example:
        >>> configure_recovery(
        ...     max_retries=5,
        ...     initial_backoff=1.0,
        ...     max_backoff=60.0,
        ...     backoff_factor=2.0,
        ...     jitter=0.2,
        ...     enabled=True,
        ...     reconnect_timeout=300.0,
        ...     auto_reconnect=True
        ... )
    """
    with _lock:
        # Construct new settings, preserving existing values for unspecified parameters
        settings = RecoverySettings(
            max_retries=max_retries or _recovery_settings["max_retries"],
            initial_backoff=initial_backoff or _recovery_settings["initial_backoff"],
            max_backoff=max_backoff or _recovery_settings["max_backoff"],
            backoff_factor=backoff_factor or _recovery_settings["backoff_factor"],
            jitter=jitter or _recovery_settings["jitter"],
            enabled=enabled if enabled is not None else _recovery_settings["enabled"],
            reconnect_timeout=reconnect_timeout
            or _recovery_settings["reconnect_timeout"],
            auto_reconnect=(
                auto_reconnect
                if auto_reconnect is not None
                else _recovery_settings["auto_reconnect"]
            ),
        )
        # Update the global settings
        _recovery_settings.update(settings._asdict())


def get_recovery_settings() -> Dict[str, Any]:
    """
    Get the current recovery settings.

    Returns:
        Dictionary containing all recovery configuration parameters
    """
    with _lock:
        return dict(_recovery_settings)


def on_reconnect(callback: Callable[[], None]) -> None:
    """
    Register a callback to be called when reconnection succeeds.

    This allows applications to react to successful reconnections, for example
    by clearing error messages, refreshing data, or notifying users.

    Callbacks are executed synchronously in the connection monitoring thread,
    so they should complete quickly to avoid blocking the monitoring process.

    Args:
        callback: Function to call on successful reconnection

    Example:
        >>> def handle_reconnection():
        ...     print("Connection restored!")
        ...     refresh_user_interface()
        >>> on_reconnect(handle_reconnection)
    """
    with _lock:
        _on_reconnect_callbacks.append(callback)


def on_disconnect(callback: Callable[[], None]) -> None:
    """
    Register a callback to be called when a disconnection is detected.

    This allows applications to react to disconnections, for example by
    displaying error messages, entering offline mode, or attempting to
    preserve unsaved data.

    Callbacks are executed synchronously in the connection monitoring thread,
    so they should complete quickly to avoid blocking the monitoring process.

    Args:
        callback: Function to call when disconnection occurs

    Example:
        >>> def handle_disconnection():
        ...     display_error_message("Connection lost. Attempting to reconnect...")
        ...     switch_to_offline_mode()
        >>> on_disconnect(handle_disconnection)
    """
    with _lock:
        _on_disconnect_callbacks.append(callback)


def is_connected() -> bool:
    """
    Check if the client is currently connected.

    Returns:
        True if connected, False otherwise
    """
    with _lock:
        return _connection_state["is_connected"]


def set_connected(connected: bool = True) -> None:
    """
    Update the connection state and trigger appropriate callbacks.

    This function serves as the central point for updating connection state. When the state
    changes, it automatically calls the appropriate callbacks and initiates reconnection
    when disconnected (if auto_reconnect is enabled).

    Args:
        connected: New connection state (True for connected, False for disconnected)
    """
    with _lock:
        # Store the previous state to detect state changes
        prev_state = _connection_state["is_connected"]
        _connection_state["is_connected"] = connected

        if connected:
            # On successful connection, update timestamp and reset reconnection counters
            _connection_state["last_connection_time"] = time.time()
            _reconnection_status["attempting"] = False
            _reconnection_status["attempts"] = 0

            # Only trigger callbacks if this is a state change from disconnected to connected
            if not prev_state:
                _execute_callbacks(_on_reconnect_callbacks, "reconnect")
        elif prev_state and not connected:
            # Handle transition from connected to disconnected
            # First trigger disconnect callbacks to notify listeners
            _execute_callbacks(_on_disconnect_callbacks, "disconnect")

            # Start automatic reconnection if enabled and not already attempting
            if (
                _recovery_settings["auto_reconnect"]
                and not _reconnection_status["attempting"]
            ):
                start_reconnection()


def _execute_callbacks(callbacks: List[Callable[[], None]], callback_type: str) -> None:
    """
    Execute a list of callback functions with error handling.

    Iterates through the provided callbacks, executing each one while catching and logging
    any exceptions that occur. This ensures that one failing callback doesn't prevent
    others from executing.

    Args:
        callbacks: List of callback functions to execute
        callback_type: Type of callback (e.g., "reconnect" or "disconnect") for logging purposes
    """
    for callback in callbacks:
        try:
            callback()
        except Exception as e:
            logger.error(f"Error in {callback_type} callback: {e}")


def calculate_backoff(attempt: int) -> float:
    """
    Calculate the backoff time for a given attempt using exponential backoff with jitter.

    This function implements an exponential backoff algorithm with random jitter to prevent
    the "thundering herd" problem, where multiple clients would retry simultaneously.

    The formula used is:
    1. Base backoff = min(max_backoff, initial_backoff * (backoff_factor ^ attempt))
    2. Jitter = base_backoff * jitter_factor
    3. Final backoff = base_backoff ± random amount of jitter

    Args:
        attempt: Current attempt number (0-based)

    Returns:
        Delay in seconds before next attempt
    """
    # Calculate exponential backoff with maximum cap
    # Formula: min(max_backoff, initial_backoff * (backoff_factor ^ attempt))
    backoff = min(
        _recovery_settings["max_backoff"],
        _recovery_settings["initial_backoff"]
        * (_recovery_settings["backoff_factor"] ** attempt),
    )

    # Add random jitter to prevent synchronized retries (thundering herd problem)
    # Jitter range is ±(backoff * jitter_factor)
    jitter_amount = backoff * _recovery_settings["jitter"]

    # Ensure minimum backoff of 0.1 seconds even with negative jitter
    return max(0.1, backoff + random.uniform(-jitter_amount, jitter_amount))


def with_retry(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """
    Execute a function with automatic retry on failure.

    This is the core retry mechanism that implements the exponential backoff strategy.
    If the function fails with any exception, it will be retried up to max_retries times
    with increasing delays between attempts, unless recovery is disabled.

    The function will be considered successful if it returns without raising an exception,
    and the connection state will be marked as connected.

    If the function fails even after all retry attempts, the last exception will be re-raised.

    Args:
        func: Function to execute with retry capability
        *args: Positional arguments for function
        **kwargs: Keyword arguments for function

    Returns:
        Result of function execution if successful

    Raises:
        Exception: Last exception if all retries fail

    Example:
        >>> def fetch_data(query):
        ...     return client.search(query)
        >>> # Retry fetch_data up to max_retries times
        >>> result = with_retry(fetch_data, "search query")
    """
    # If recovery is disabled, just execute the function normally without retry logic
    if not _recovery_settings["enabled"]:
        return func(*args, **kwargs)

    last_exception = None
    # Try the function up to max_retries + 1 times (initial attempt + retries)
    for attempt in range(_recovery_settings["max_retries"] + 1):
        try:
            # Execute the function with provided arguments
            result = func(*args, **kwargs)
            # Success - mark as connected and return the result
            set_connected(True)
            return cast(T, result)
        except Exception as e:
            # Store the exception for potential re-raising later
            last_exception = e
            # Mark as disconnected since the operation failed
            set_connected(False)

            # If this was the last allowed attempt, re-raise the exception
            if attempt >= _recovery_settings["max_retries"]:
                raise

            # Calculate backoff time with jitter to prevent thundering herd
            backoff = calculate_backoff(attempt)
            logger.warning(
                f"Operation failed (attempt {attempt+1}/{_recovery_settings['max_retries']+1}): "
                f"{e}. Retrying in {backoff:.2f} seconds..."
            )
            # Wait before next attempt using calculated backoff time
            time.sleep(backoff)

    # This should only execute if max_retries < 0, which shouldn't occur in normal operation
    # Still handled as a safety measure for unexpected configuration values
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry failed for unknown reason")


def start_reconnection() -> None:
    """
    Start a background thread to attempt reconnection to the server.

    This function initiates the reconnection process in a separate daemon thread,
    allowing the main application to continue functioning while reconnection is
    attempted in the background.

    The reconnection thread will make repeated attempts with increasing delays
    until either successful reconnection or the reconnect_timeout is reached.

    Only starts if not already attempting and server URL is configured. Uses a daemon
    thread to ensure it doesn't prevent program exit if the application terminates.

    Example:
        >>> # Manual invocation, usually not needed as it's automatic
        >>> if not is_connected():
        ...     start_reconnection()
    """
    with _lock:
        # Skip if already attempting reconnection or no server URL configured
        if _reconnection_status["attempting"] or not _connection_state["server_url"]:
            return

        # Reset reconnection status for a new attempt sequence
        _reconnection_status.update(
            {"attempting": True, "attempts": 0, "last_attempt": time.time()}
        )

        # Create and start daemon thread for reconnection attempts
        # Using daemon=True ensures the thread won't block program exit
        thread = threading.Thread(
            target=_reconnection_worker, name="FAISSx-Reconnect", daemon=True
        )
        _reconnection_status["thread"] = thread
        thread.start()


def _reconnection_worker() -> None:
    """
    Background worker thread function for reconnection attempts.

    This is an internal function executed in a background thread that performs
    the actual reconnection logic. It will continue making attempts until either:
    1. Successful reconnection
    2. The reconnect_timeout period is exceeded
    3. The attempting flag is set to False (via cancel_recovery)

    Each attempt follows these steps:
    1. Get current connection parameters
    2. Calculate backoff time based on attempt number
    3. Try to connect with the stored parameters
    4. If successful, update connection state and exit
    5. If unsuccessful, wait for backoff period and try again

    Implements exponential backoff with jitter for reconnection attempts.
    """
    # Record start time to enforce the overall timeout limit
    start_time = time.time()

    # Continue attempts while within timeout and still attempting
    while (
        _reconnection_status["attempting"]
        and time.time() - start_time < _recovery_settings["reconnect_timeout"]
    ):
        try:
            # Get current connection parameters and attempt count under lock to ensure thread safety
            with _lock:
                server_url = _connection_state["server_url"]
                api_key = _connection_state["api_key"]
                tenant_id = _connection_state["tenant_id"]
                attempt = _reconnection_status["attempts"]
                # Increment attempt counter for the next iteration
                _reconnection_status["attempts"] += 1
                # Update timestamp of last attempt for monitoring
                _reconnection_status["last_attempt"] = time.time()

            # Calculate backoff time with jitter for this attempt
            backoff = calculate_backoff(attempt)
            logger.info(
                f"Attempting reconnection (attempt {attempt+1}). Server: {server_url}"
            )

            # First check if we can reuse existing client
            client = get_client()
            if client is None or not client.ping():
                # Existing client doesn't work, need to reconfigure
                from . import client as client_module

                # Reconfigure client with stored parameters from previous configuration
                client_module.configure(
                    server=server_url, api_key=api_key, tenant_id=tenant_id
                )
                # Get the newly configured client
                client = get_client()

                # Verify reconnection success with ping check
                if client and client.ping():
                    logger.info(f"Reconnection successful after {attempt+1} attempts")
                    # Update state and notify callbacks
                    set_connected(True)
                    return  # Exit the reconnection worker on success

            # If we reached here, reconnection attempt failed
            logger.info(
                f"Reconnection attempt failed. Retrying in {backoff:.2f} seconds..."
            )
            # Wait before next attempt using calculated backoff time
            time.sleep(backoff)

        except Exception as e:
            # Handle any unexpected exceptions during reconnection attempts
            logger.error(f"Error during reconnection attempt: {e}")
            # Brief delay before next attempt when unexpected error occurs
            time.sleep(1.0)

    # If we reach here, we've exhausted retries or timeout was reached
    with _lock:
        # Mark reconnection as complete
        _reconnection_status["attempting"] = False
        logger.warning(
            f"Reconnection attempts exhausted after {_reconnection_status['attempts']} "
            f"tries. Giving up."
        )


def retry_operation(func_name: str, *args: Any, **kwargs: Any) -> Any:
    """
    Retry a client operation with the configured retry policy.

    This is a convenience wrapper around with_retry() that handles getting the client and
    calling the specified method dynamically. It provides an easier interface for retrying
    operations without directly handling the client.

    The function gets the current client, ensures it exists, and then applies the retry
    mechanism to the specified client method with the provided arguments.

    Args:
        func_name: Name of the client method to call (e.g. "search", "add", "ping")
        *args: Positional arguments for the method
        **kwargs: Keyword arguments for the method

    Returns:
        Result of the operation if successful

    Raises:
        RuntimeError: If client is not configured or available

    Example:
        >>> # Retry a search operation with exponential backoff
        >>> results = retry_operation("search", query_vector, k=10)
    """
    # Get the current client instance
    client = get_client()
    if client is None:
        # Cannot retry without a client
        raise RuntimeError("FAISSx client not available - please configure first")

    # Dynamically get the method by name and apply retry mechanism
    return with_retry(getattr(client, func_name), *args, **kwargs)


def is_recovering() -> bool:
    """
    Check if reconnection is currently being attempted.

    Returns:
        True if reconnection is in progress, False otherwise
    """
    with _lock:
        return _reconnection_status["attempting"]


def last_attempt_time() -> float:
    """
    Get the timestamp of the last reconnection attempt.

    Returns:
        Unix timestamp of last attempt
    """
    with _lock:
        return _reconnection_status["last_attempt"]


def attempts_count() -> int:
    """
    Get the number of reconnection attempts made.

    Returns:
        Total number of reconnection attempts
    """
    with _lock:
        return _reconnection_status["attempts"]


def cancel_recovery() -> None:
    """
    Cancel any ongoing reconnection attempts.

    Sets the attempting flag to False, which will cause the reconnection worker
    to exit on its next iteration.
    """
    with _lock:
        _reconnection_status["attempting"] = False


def force_reconnect() -> None:
    """
    Force an immediate reconnection attempt.

    Marks the connection as disconnected and starts a new reconnection attempt
    regardless of current state.
    """
    set_connected(False)
    start_reconnection()


def _store_connection_params(
    server: str, api_key: Optional[str], tenant_id: Optional[str]
) -> None:
    """
    Store connection parameters for future reconnection attempts.

    Called by the client module when configure() is called to maintain connection
    state for recovery purposes. This information is used by the reconnection worker
    to reestablish the connection with the same parameters when disconnection occurs.

    Updates the connection state to indicate active connection and records the
    connection time.

    Args:
        server: Server URL to connect to
        api_key: Optional API key for authentication
        tenant_id: Optional tenant ID for multi-tenant setups
    """
    with _lock:
        _connection_state["server_url"] = server
        _connection_state["api_key"] = api_key
        _connection_state["tenant_id"] = tenant_id
        _connection_state["is_connected"] = True
        _connection_state["last_connection_time"] = time.time()


def _start_connection_monitor() -> None:
    """
    Start a background thread to monitor the connection status.

    This internal function initializes and starts the connection monitoring thread,
    which periodically checks if the connection to the FAISSx server is still active
    and initiates reconnection if needed.

    The monitor runs as a daemon thread that:
    1. Periodically wakes up and checks connection health
    2. Triggers reconnection process when a disconnection is detected
    3. Updates connection state when server responds

    Since this is a daemon thread, it won't prevent program exit when the main
    program terminates.
    """

    def monitor_loop() -> None:
        """
        Main monitoring loop that checks connection status periodically.

        This loop runs continuously in the background as a daemon thread and is responsible for:
        1. Detecting disconnections by pinging the server
        2. Updating connection state based on ping results
        3. Initiating reconnection when disconnections are detected

        The monitor sleeps for 30 seconds between checks to avoid excessive network traffic.
        It skips checks if recovery is disabled, not configured, or if reconnection is already
        in progress to avoid conflicting reconnection attempts.

        The function catches all exceptions to ensure the monitoring thread continues running
        regardless of transient errors.
        """
        while True:
            try:
                # Sleep first to allow initial connection time and avoid immediate checking
                time.sleep(30.0)  # Check every 30 seconds

                # Skip monitoring if recovery is globally disabled in settings
                if not _recovery_settings["enabled"]:
                    continue

                # Skip if no server URL is configured (client not initialized yet)
                if not _connection_state["server_url"]:
                    continue

                # Skip if reconnection is already in progress to avoid conflicting recovery attempts
                if _reconnection_status["attempting"]:
                    continue

                # Try to get the client and check the connection with a ping
                client = get_client()
                if client:
                    try:
                        # Ping server to verify connection is alive
                        client.ping()
                        # If ping succeeds, update connection state to connected
                        set_connected(True)
                    except Exception:
                        # Ping failed - mark as disconnected which will trigger reconnection
                        # if auto_reconnect is enabled in settings
                        set_connected(False)
            except Exception as e:
                # Catch any unexpected exceptions to keep the monitor alive and running
                logger.error(f"Error in connection monitor: {e}")

    # Start the monitor thread as a daemon to allow program exit when the main thread exits
    thread = threading.Thread(target=monitor_loop, name="FAISSx-Monitor", daemon=True)
    thread.start()


# Initialize connection monitoring when module is imported
_start_connection_monitor()
