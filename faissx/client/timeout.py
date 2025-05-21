#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Timeout decorator for enforcing operation timeouts
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
Timeout decorator module for FAISSx client.

This module provides a decorator that enforces timeouts on function calls
to prevent operations from blocking indefinitely. It allows for global,
per-function, and per-instance timeout configuration, and gracefully
handles the interruption of long-running operations.

Key features:
- Configurable timeouts at global, decorator, and instance level
- Customizable exception type for timeout errors
- Performance monitoring and logging for slow operations
- Thread-safe behavior for timeout management

Architecture:
The timeout mechanism works by creating a separate timer thread that will
interrupt the main thread if the operation takes longer than the specified time.
This avoids the need for signal handling and works reliably across platforms.
"""

import threading
import functools
import time
import logging
from typing import Callable, Optional, Type, TypeVar, Any, cast, Union, overload

# Configure logging for the module
logger = logging.getLogger(__name__)

# Define thread-safe lock for global timeout state
# This ensures consistent updates to the timeout configuration across multiple threads
_timeout_lock = threading.RLock()

# Global timeout in seconds - default timeout for all operations
# This value is used when no specific timeout is provided to the decorator
TIMEOUT = 5.0

# Generic return type for decorated functions
R = TypeVar('R')  # Used to preserve the original return type of decorated functions
T = TypeVar('T')  # For the class type in method calls


class TimeoutError(RuntimeError):
    """
    Exception raised when a function call exceeds its time limit.

    This is a custom subclass of RuntimeError that provides
    clearer error messages and improved traceability for
    timeout-related issues.

    Usage:
        This exception is automatically raised by the timeout decorator
        when a function takes longer than its allowed execution time.
    """
    pass


def set_timeout(timeout: float) -> None:
    """
    Set the global default timeout value for all operations.

    This function updates the global timeout used when no specific
    timeout is provided to the decorator or available on the instance.
    The change affects all future operations using the default timeout.

    Args:
        timeout: New timeout value in seconds (must be positive)

    Raises:
        ValueError: If timeout value is negative or zero

    Example:
        >>> set_timeout(10.0)  # Set global timeout to 10 seconds
        >>> @timeout  # Uses the new global timeout
        ... def slow_operation():
        ...     pass
    """
    if timeout <= 0:
        raise ValueError("Timeout value must be positive")

    global TIMEOUT
    # Use a lock to safely update the global timeout value
    # This prevents race conditions when multiple threads try to update it simultaneously
    with _timeout_lock:
        TIMEOUT = timeout


def interrupt_function(func_name: str, exception_cls: Type[Exception] = TimeoutError) -> None:
    """
    Interrupt the main thread with a timeout exception.

    This function finds the main thread and raises the specified exception in it,
    effectively interrupting the current operation. It handles thread state locks
    appropriately for different Python versions.

    Args:
        func_name: Name of the function that timed out (for error message)
        exception_cls: Exception class to raise on timeout

    Raises:
        The specified exception in the main thread

    Implementation Note:
        This approach works by finding the main thread and releasing its state lock
        before raising an exception. This technique requires careful handling of
        Python's thread synchronization mechanisms.
    """
    # Create a descriptive error message that includes the function name
    error_msg = f"Operation {func_name} timed out"
    logger.error(error_msg)

    # Find the main thread by its name from all active threads
    # We use next() with a generator expression and provide a None fallback
    main_thread = next(
        (t for t in threading.enumerate() if t.name == "MainThread"),
        None
    )

    # If we couldn't find the main thread, log a warning and exit
    if main_thread is None:
        logger.warning("Could not find main thread for interruption")
        return

    # Handle thread state lock for Python 3.7+
    # The _tstate_lock must be released before we can raise an exception in the thread
    if hasattr(main_thread, "_tstate_lock"):
        if main_thread._tstate_lock:  # type: ignore
            try:
                main_thread._tstate_lock.release()  # type: ignore
            except (RuntimeError, AttributeError) as e:
                # This could happen if the lock was already released or during interpreter shutdown
                logger.debug(f"Failed to release thread state lock: {e}")

    # Raise the exception to interrupt the operation
    # This will propagate to the main thread and terminate the long-running function
    raise exception_cls(error_msg)


# Type signature for decorator applied with explicit timeout parameter
@overload
def timeout(
    seconds: float,
    exception_cls: Type[Exception] = TimeoutError
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    ...


# Type signature for decorator applied directly to a function
@overload
def timeout(func: Callable[..., R]) -> Callable[..., R]:
    ...


def timeout(
    seconds_or_func: Union[Optional[float], Callable[..., Any]] = None,
    exception_cls: Type[Exception] = TimeoutError
) -> Union[Callable[[Callable[..., R]], Callable[..., R]], Callable[..., R]]:
    """
    Decorator that enforces a timeout on function execution.

    If the decorated function takes longer than the specified timeout,
    it will be interrupted and the specified exception will be raised.
    The decorator supports three levels of timeout configuration:

    1. Explicitly specified in the decorator
    2. From instance attribute (self.timeout) if available
    3. Global default timeout as fallback

    Args:
        seconds_or_func: Either the timeout in seconds or the function to decorate
                         (if None, reads from instance or uses default)
        exception_cls: Exception class to raise on timeout

    Returns:
        Decorated function with timeout enforcement

    Usage:
        # With explicit timeout:
        @timeout(5.0)
        def potentially_slow_function():
            ...

        # With instance-level timeout:
        @timeout()
        def method_with_instance_timeout(self):
            # Will use self.timeout
            ...

        # As a simple decorator:
        @timeout
        def use_global_timeout():
            # Will use global TIMEOUT
            ...

    Implementation Details:
        The decorator creates a timer thread that will trigger after the specified timeout.
        If the function completes before the timeout, the timer is cancelled.
        Otherwise, the timer will raise an exception in the main thread.
    """
    # Handle case where timeout is used as @timeout without parentheses
    # In this case, seconds_or_func is actually the function being decorated
    if callable(seconds_or_func):
        func = seconds_or_func
        seconds = None

        @functools.wraps(func)
        def direct_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Delegate to the internal implementation
            return _timeout_wrapper(func, None, exception_cls, args, kwargs)

        return direct_wrapper

    # Normal case with seconds specified or defaulted
    seconds = seconds_or_func

    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        """
        Inner decorator function that wraps the target function.

        This preserves the original function's name, docstring, and signature.
        """
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            # Cast is necessary to maintain the correct return type
            return cast(R, _timeout_wrapper(func, seconds, exception_cls, args, kwargs))

        return wrapper

    return decorator


def _timeout_wrapper(
    func: Callable[..., Any],
    seconds: Optional[float],
    exception_cls: Type[Exception],
    args: Any,
    kwargs: Any
) -> Any:
    """
    Internal implementation of the timeout wrapper.

    This function handles the actual timeout logic, separated from the decorator
    interface for clarity. It determines the appropriate timeout value to use,
    creates a timer thread, executes the function, and cleans up afterward.

    Args:
        func: The function being wrapped
        seconds: Timeout value from decorator, if provided
        exception_cls: Exception class to raise on timeout
        args: Positional arguments for the wrapped function
        kwargs: Keyword arguments for the wrapped function

    Returns:
        Result of the wrapped function

    Notes:
        This wrapper performs several key tasks:
        1. Resolves the timeout value from multiple possible sources
        2. Sets up a timer thread to interrupt execution if needed
        3. Monitors execution time and logs slow operations
        4. Ensures proper cleanup of resources
    """
    # Get timeout value from decorator arg, instance, or global default
    timeout_value = seconds

    # Check for instance-level timeout if first arg is self (instance method)
    # This allows classes to define their own timeout values
    if timeout_value is None and args and hasattr(args[0], 'timeout'):
        timeout_value = args[0].timeout

    # Fall back to global default if no timeout specified
    if timeout_value is None:
        with _timeout_lock:
            timeout_value = TIMEOUT

    # Ensure timeout is positive to avoid immediate timeouts or errors
    if timeout_value <= 0:
        logger.warning(
            f"Non-positive timeout value {timeout_value} for {func.__qualname__}, "
            f"using default"
        )
        with _timeout_lock:
            timeout_value = TIMEOUT

    # Define timeout handler to interrupt execution
    # This function will be called by the timer thread when the timeout is reached
    def handle_timeout() -> None:
        """Inner function that is triggered when timeout occurs."""
        interrupt_function(func.__qualname__, exception_cls)

    # Create and start daemon timer thread
    # Using a daemon thread ensures it won't prevent program exit
    timer = threading.Timer(timeout_value, handle_timeout)
    timer.daemon = True
    timer.start()

    # Record start time to measure function execution duration
    start_time = time.time()
    try:
        # Execute the wrapped function with its original arguments
        result = func(*args, **kwargs)

        # Calculate elapsed time after function completes
        elapsed = time.time() - start_time

        # Log warning for operations using >80% of timeout
        # This helps identify functions that are close to timing out
        if elapsed > timeout_value * 0.8:
            logger.warning(
                f"{func.__qualname__} took {elapsed:.2f}s "
                f"(timeout: {timeout_value:.2f}s)"
            )
        return result
    finally:
        # Always clean up timer to prevent resource leaks
        # This ensures the timer thread doesn't trigger after function completion
        timer.cancel()
