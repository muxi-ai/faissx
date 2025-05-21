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
to prevent operations from blocking indefinitely.
"""

import threading
import functools
import time
import logging
from typing import Callable, Optional, Type

# Configure logging for the module
logger = logging.getLogger(__name__)

# Global timeout in seconds
TIMEOUT = 5.0


class TimeoutError(RuntimeError):
    """Exception raised when a function call times out."""
    pass


def set_timeout(timeout: float):
    global TIMEOUT
    TIMEOUT = timeout


def interrupt_function(func_name: str, exception_cls: Type[Exception] = TimeoutError):
    """
    Interrupt the main thread with a timeout exception.

    Args:
        func_name: Name of the function that timed out
        exception_cls: Exception class to raise

    Raises:
        The specified exception in the main thread
    """
    import threading
    error_msg = f"Operation {func_name} timed out"
    logger.error(error_msg)

    # Raise the exception in the main thread
    main_thread = next(
        t for t in threading.enumerate()
        if t.name == "MainThread"
    )
    if hasattr(main_thread, "_tstate_lock"):
        # Python 3.7+ thread state handling
        if main_thread._tstate_lock:  # type: ignore
            main_thread._tstate_lock.release()  # type: ignore

    raise exception_cls(error_msg)


def timeout(seconds: Optional[float] = None, exception_cls: Type[Exception] = TimeoutError):
    """
    Decorator that enforces a timeout on function execution.

    If the decorated function takes longer than the specified timeout,
    it will be interrupted and the specified exception will be raised.

    Args:
        seconds: Timeout in seconds (if None, reads from instance or uses default)
        exception_cls: Exception class to raise on timeout

    Returns:
        Decorated function with timeout enforcement

    Usage:
        @timeout(5.0)
        def potentially_slow_function():
            ...

        # Or with a method that has self.timeout
        @timeout()
        def method_with_instance_timeout(self):
            # Will use self.timeout
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Determine the actual timeout value to use
            timeout_value = seconds

            # If no timeout specified, try to get it from the instance (first arg)
            if timeout_value is None and args and hasattr(args[0], 'timeout'):
                timeout_value = args[0].timeout

            # Default timeout if still not specified
            if timeout_value is None:
                timeout_value = TIMEOUT

            # Function to raise the exception
            def handle_timeout():
                interrupt_function(func.__qualname__, exception_cls)

            # Set up the timer
            timer = threading.Timer(timeout_value, handle_timeout)
            timer.daemon = True
            timer.start()

            start_time = time.time()
            try:
                # Call the wrapped function
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time

                # Log slow operations (taking more than 80% of allowed time)
                if elapsed > timeout_value * 0.8:
                    logger.warning(
                        f"{func.__qualname__} took {elapsed:.2f}s "
                        f"(timeout: {timeout_value:.2f}s)"
                    )
                return result
            finally:
                # Always cancel the timer to avoid dangling threads
                timer.cancel()

        return wrapper

    return decorator
