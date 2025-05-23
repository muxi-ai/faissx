#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FAISSx Server Logging
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
Enterprise Logging Configuration for FAISSx Server

This module provides comprehensive logging configuration and utilities for the FAISSx
server infrastructure. It supports both console and file-based logging with configurable
formats, levels, and output destinations for production and development environments.

Key Features:
- Structured logging configuration with multiple output handlers
- Predefined format templates for different deployment scenarios
- Automatic log level validation and error handling
- Console and file logging with configurable formatters
- Production-ready logging patterns with timestamps and context

Usage:
- Development: Simple console logging with readable format
- Production: Structured file logging with detailed timestamps
- Debug: Verbose logging with full context information

Integration:
This module integrates seamlessly with the FAISSx server components to provide
consistent logging across all modules including authentication, indexing, search,
and hybrid operations.
"""

import logging
import logging.handlers
import sys
from typing import Optional
from pathlib import Path

# Logging configuration constants for consistent formatting across the system
# These constants provide predefined templates for different deployment scenarios

# Standard log formats for different environments
LOG_FORMAT_SIMPLE = "%(levelname)s - %(message)s"
LOG_FORMAT_STANDARD = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FORMAT_DETAILED = (
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)
LOG_FORMAT_PRODUCTION = (
    "%(asctime)s - %(name)s - %(levelname)s - %(process)d - "
    "%(filename)s:%(lineno)d - %(funcName)s - %(message)s"
)

# Date formats for log timestamps
DATE_FORMAT_STANDARD = "%Y-%m-%d %H:%M:%S"
DATE_FORMAT_ISO = "%Y-%m-%dT%H:%M:%S"
DATE_FORMAT_COMPACT = "%m%d %H:%M:%S"

# Valid log levels for validation
VALID_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}

# Default configuration values
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB default max file size
DEFAULT_BACKUP_COUNT = 5  # Keep 5 backup files by default

# Set up the package-level logger with proper hierarchy
logger = logging.getLogger("faissx")


def _validate_log_level(log_level: str) -> str:
    """
    Validate and normalize the log level string.

    Args:
        log_level: Log level string to validate

    Returns:
        str: Validated and normalized log level

    Raises:
        ValueError: If the log level is invalid
    """
    normalized_level = log_level.upper().strip()
    if normalized_level not in VALID_LOG_LEVELS:
        raise ValueError(
            f"Invalid log level '{log_level}'. "
            f"Valid levels are: {', '.join(sorted(VALID_LOG_LEVELS))}"
        )
    return normalized_level


def _create_formatter(
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
    use_colors: bool = False,
) -> logging.Formatter:
    """
    Create a logging formatter with specified format and date format.

    Args:
        log_format: Custom log format string (uses standard format if None)
        date_format: Date format string (uses standard format if None)
        use_colors: Whether to use colored output (for console logging)

    Returns:
        logging.Formatter: Configured formatter instance
    """
    if log_format is None:
        log_format = LOG_FORMAT_STANDARD

    if date_format is None:
        date_format = DATE_FORMAT_STANDARD

    return logging.Formatter(log_format, datefmt=date_format)


def configure_logging(
    log_level: str = DEFAULT_LOG_LEVEL,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
    console_output: bool = True,
    file_rotation: bool = False,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    backup_count: int = DEFAULT_BACKUP_COUNT,
) -> None:
    """
    Configure comprehensive logging for the FAISSx server with validation and options.

    This function sets up logging with multiple output destinations, format options,
    and production-ready features like file rotation. It replaces any existing
    logging configuration to ensure consistent behavior.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL, NOTSET)
        log_file: Path to log file (None for console only logging)
        log_format: Custom log format string (None for standard format)
        date_format: Custom date format string (None for standard format)
        console_output: Whether to include console output handler
        file_rotation: Whether to use rotating file handler instead of basic file handler
        max_file_size: Maximum file size in bytes before rotation (default: 10MB)
        backup_count: Number of backup files to keep when rotating (default: 5)

    Raises:
        ValueError: If log_level is invalid
        OSError: If log_file path is invalid or not writable

    Example:
        >>> # Basic console logging
        >>> configure_logging(log_level="DEBUG")

        >>> # Production file logging with rotation
        >>> configure_logging(
        ...     log_level="INFO",
        ...     log_file="/var/log/faissx/server.log",
        ...     file_rotation=True,
        ...     max_file_size=50*1024*1024  # 50MB
        ... )
    """
    # Validate and normalize log level
    try:
        validated_level = _validate_log_level(log_level)
        numeric_level = getattr(logging, validated_level)
    except ValueError:
        logger.warning(
            f"Invalid log level '{log_level}', using default '{DEFAULT_LOG_LEVEL}'"
        )
        numeric_level = getattr(logging, DEFAULT_LOG_LEVEL)
        validated_level = DEFAULT_LOG_LEVEL

    # Configure root logger and clear existing handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove any existing handlers to avoid duplication
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter with specified options
    formatter = _create_formatter(log_format, date_format)

    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        logger.debug("Console logging handler configured")

    # Add file handler if log file is specified
    if log_file:
        try:
            # Create directory if it doesn't exist
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Choose between rotating and regular file handler
            if file_rotation:
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=max_file_size,
                    backupCount=backup_count,
                    encoding="utf-8",
                )
                logger.debug(
                    f"Rotating file handler configured: {log_file} "
                    f"(max: {max_file_size} bytes, backups: {backup_count})"
                )
            else:
                file_handler = logging.FileHandler(log_file, encoding="utf-8")
                logger.debug(f"File handler configured: {log_file}")

            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        except (OSError, IOError) as e:
            logger.error(f"Failed to configure file logging to '{log_file}': {e}")
            logger.warning("Continuing with console logging only")

    # Log configuration summary
    logger.info(
        f"Logging configured - Level: {validated_level}, "
        f"Console: {console_output}, File: {log_file is not None}"
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified module or component.

    This function provides a consistent way to create logger instances
    across the FAISSx codebase with proper naming hierarchy.

    Args:
        name: Name of the logger (typically __name__ or module name)

    Returns:
        logging.Logger: Configured logger instance

    Example:
        >>> # In a module
        >>> module_logger = get_logger(__name__)
        >>> module_logger.info("Module started")
    """
    return logging.getLogger(name)


def set_module_log_level(module_name: str, log_level: str) -> None:
    """
    Set the log level for a specific module or component.

    This allows fine-grained control over logging levels for different
    parts of the FAISSx system without affecting the global configuration.

    Args:
        module_name: Name of the module (e.g., "faissx.server.auth")
        log_level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Raises:
        ValueError: If log_level is invalid

    Example:
        >>> # Enable debug logging for authentication module only
        >>> set_module_log_level("faissx.server.auth", "DEBUG")
    """
    validated_level = _validate_log_level(log_level)
    numeric_level = getattr(logging, validated_level)

    module_logger = logging.getLogger(module_name)
    module_logger.setLevel(numeric_level)

    logger.debug(f"Set log level for '{module_name}' to {validated_level}")


def configure_development_logging() -> None:
    """
    Configure logging optimized for development environment.

    Sets up simple, readable console logging with detailed format
    suitable for debugging and development work.
    """
    configure_logging(
        log_level="DEBUG",
        log_format=LOG_FORMAT_DETAILED,
        console_output=True,
        file_rotation=False,
    )
    logger.info("Development logging configuration applied")


def configure_production_logging(
    log_file: str,
    log_level: str = "INFO",
    max_file_size: int = 50 * 1024 * 1024,  # 50MB
) -> None:
    """
    Configure logging optimized for production environment.

    Sets up structured file logging with rotation, minimal console output,
    and production-ready formatting with full context information.

    Args:
        log_file: Path to the main log file
        log_level: Logging level for production (default: INFO)
        max_file_size: Maximum file size before rotation (default: 50MB)
    """
    configure_logging(
        log_level=log_level,
        log_file=log_file,
        log_format=LOG_FORMAT_PRODUCTION,
        console_output=False,
        file_rotation=True,
        max_file_size=max_file_size,
        backup_count=10,  # Keep more backups in production
    )
    logger.info(f"Production logging configuration applied: {log_file}")
