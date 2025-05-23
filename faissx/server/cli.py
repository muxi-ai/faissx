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
FAISSx Server CLI

Command-line interface for the FAISSx server. This module provides a comprehensive CLI
for running and managing the FAISSx vector database proxy server, including configuration
of authentication, data persistence, network settings, and logging.

The CLI supports multiple commands and provides extensive configuration options for
production deployments. It handles argument parsing, validation, server configuration,
and graceful error handling.

Key Features:
- Server startup and configuration management
- Authentication setup (API keys, JSON files)
- Network configuration (ports, bind addresses)
- Data persistence configuration
- Logging level control
- Version information display

Usage:
    faissx-server run --port 8080 --enable-auth --auth-keys "key1:tenant1"
    faissx-server --version
"""

import sys                          # System-specific parameters and functions
import argparse                     # Command-line argument parsing library
from typing import Dict, Optional   # Type hints for better code documentation
from faissx import server          # Main server module for FAISSx operations
from faissx import __version__     # Version information for display
import logging                     # Logging framework for operational visibility

# Constants for CLI configuration
# These constants centralize default values and make the code more maintainable
DEFAULT_PORT = 45678               # Default server port (chosen to avoid conflicts)
DEFAULT_BIND_ADDRESS = "0.0.0.0"  # Default bind address (all network interfaces)
DEFAULT_LOG_LEVEL = "WARNING"     # Default logging level (production-friendly)
SUCCESS_EXIT_CODE = 0             # Standard Unix exit code for successful operations
ERROR_EXIT_CODE = 1               # Standard Unix exit code for error conditions

# Valid log levels (case-insensitive)
# Supports both uppercase and lowercase variants for user convenience
# These match Python's standard logging levels exactly
VALID_LOG_LEVELS = [
    "DEBUG",        # Most verbose: all internal operations and debugging info
    "INFO",         # General information about server operations and events
    "WARNING",      # Warning messages for unusual but handled conditions
    "ERROR",        # Error messages for failures that don't crash the server
    "CRITICAL",     # Critical failures that may cause server shutdown
    "debug",        # Lowercase variants for user convenience
    "info",         # (internally converted to uppercase by logging module)
    "warning",      # These provide flexibility in user input
    "error",        # while maintaining standard logging compatibility
    "critical",     # All levels are valid Python logging levels
]

# Create logger instance for this module
# Uses hierarchical naming to integrate with FAISSx logging system
# All log messages from CLI operations will be tagged with "faissx.server"
logger = logging.getLogger("faissx.server")


def run_command(args: argparse.Namespace) -> int:
    """
    Run the FAISSx server with the specified command-line arguments.

    This function orchestrates the complete server startup process, handling configuration
    validation, authentication setup, and server lifecycle management. It provides
    comprehensive error handling and graceful shutdown capabilities.

    Process Flow:
        1. Parse and validate authentication configuration
        2. Validate configuration parameters
        3. Configure the server with all provided settings
        4. Start the server and handle runtime events
        5. Manage graceful shutdown on interruption

    Authentication Handling:
        - Parses CLI auth keys in "key1:tenant1,key2:tenant2" format
        - Validates mutual exclusivity of auth methods
        - Supports both inline keys and JSON file authentication

    Args:
        args: Command-line arguments parsed by argparse containing all CLI options
              including port, bind_address, auth configuration, data_dir, and log_level

    Returns:
        int: Exit code indicating operation result:
             - 0: Successful startup and shutdown
             - 1: Configuration error or startup failure

    Raises:
        No exceptions are raised - all errors are handled gracefully with appropriate
        logging and error exit codes.

    Example:
        >>> import argparse
        >>> args = argparse.Namespace(port=8080, enable_auth=True, ...)
        >>> exit_code = run_command(args)
    """
    # Parse and validate API keys from command line if provided
    # This handles the --auth-keys argument which expects a comma-separated list
    # Format: "key1:tenant1,key2:tenant2" -> {"key1": "tenant1", "key2": "tenant2"}
    auth_keys: Optional[Dict[str, str]] = None
    if args.auth_keys:
        # Initialize dictionary to store parsed key-tenant mappings
        auth_keys = {}
        try:
            # Split the input string on commas to get individual key:tenant pairs
            # Example: "api123:tenant1,xyz789:tenant2" -> ["api123:tenant1", "xyz789:tenant2"]
            key_pairs = args.auth_keys.split(",")

            # Process each key:tenant pair individually for validation and parsing
            for key_pair in key_pairs:
                # Remove leading/trailing whitespace from each pair to handle spacing
                # This allows for flexible input like "key1:tenant1, key2:tenant2"
                clean_pair = key_pair.strip()

                # Validate that the pair contains a colon separator
                # This ensures the required key:tenant format is followed
                if ":" not in clean_pair:
                    raise ValueError(
                        f"Invalid key pair format: '{clean_pair}' (expected 'key:tenant')"
                    )

                # Split on first colon only to handle tenant IDs that might contain colons
                # Using split(":", 1) prevents issues with tenant IDs like "org:team:user"
                api_key, tenant_id = clean_pair.split(":", 1)

                # Clean up any additional whitespace around the key and tenant
                # This handles cases like "key : tenant" or "key: tenant "
                api_key = api_key.strip()
                tenant_id = tenant_id.strip()

                # Validate that both components are non-empty after cleaning
                # Empty keys or tenants would create security vulnerabilities
                if not api_key or not tenant_id:
                    raise ValueError(f"Empty key or tenant in pair: '{clean_pair}'")

                # Store the validated key-tenant mapping in our dictionary
                # This will be used for authentication during server operations
                auth_keys[api_key] = tenant_id

            # Log successful parsing for debugging and operational visibility
            logger.info(f"Parsed {len(auth_keys)} API key pairs from command line")

        except Exception as e:
            # Provide detailed error information to help users fix their input
            logger.error(f"Error parsing API keys: {e}")
            logger.error("Expected format: --auth-keys 'key1:tenant1,key2:tenant2'")
            return ERROR_EXIT_CODE

    # Validate authentication configuration - ensure only one method is used
    # This prevents ambiguous configurations that could lead to security issues
    # The server should have a clear, unambiguous authentication source
    if args.auth_keys and args.auth_file:
        logger.error("Error: Cannot provide both --auth-keys and --auth-file")
        logger.error("Please use either --auth-keys OR --auth-file, not both")
        return ERROR_EXIT_CODE

    # Configure server with all provided settings
    # The server.configure() function handles parameter validation and internal setup
    # This is where all CLI arguments get translated into server configuration
    try:
        logger.info("Configuring FAISSx server with provided settings")

        # Pass all configuration parameters to the server module
        # The server module will validate these parameters and set up internal state
        server.configure(
            port=args.port,                    # Network port for ZeroMQ binding
            bind_address=args.bind_address,    # Network interface to bind to
            auth_keys=auth_keys,               # Parsed API key dictionary (if provided)
            auth_file=args.auth_file,          # Path to JSON auth file (if provided)
            enable_auth=args.enable_auth,      # Boolean flag to enable authentication
            data_dir=args.data_dir,            # Directory for persistent storage (optional)
            log_level=args.log_level,          # Logging verbosity level
        )

        # Log configuration summary for debugging and operational visibility
        # This helps operators verify the server is configured as expected
        logger.info(
            f"Server configured: port={args.port}, bind={args.bind_address}, "
            f"auth_enabled={args.enable_auth}, data_dir={args.data_dir}"
        )

    except ValueError as e:
        # Handle configuration validation errors (invalid parameters, etc.)
        # These are typically user input errors that can be corrected
        logger.error(f"Error configuring server: {e}")
        return ERROR_EXIT_CODE
    except Exception as e:
        # Handle unexpected configuration errors (system issues, import problems, etc.)
        # These might indicate deeper system or installation problems
        logger.error(f"Unexpected error during server configuration: {e}")
        return ERROR_EXIT_CODE

    # Start server and handle runtime events
    # This is the main server execution phase where the server begins accepting requests
    try:
        logger.info("Starting FAISSx server...")

        # Start the server main loop - this call blocks until server shutdown
        # The server will bind to the configured port and begin accepting ZeroMQ connections
        # All vector database operations (add, search, delete) happen in this loop
        server.run()

        # If we reach this point, the server has shut down cleanly
        logger.info("Server shut down successfully")
        return SUCCESS_EXIT_CODE

    except KeyboardInterrupt:
        # Handle Ctrl+C (SIGINT) gracefully - this is expected user behavior
        # The server should shut down cleanly when the user presses Ctrl+C
        # This is not an error condition, just normal administrative shutdown
        logger.info("\nServer stopped by user (Ctrl+C)")
        return SUCCESS_EXIT_CODE

    except Exception as e:
        # Handle any unexpected runtime errors that occur during server operation
        # These could include network errors, resource exhaustion, or internal bugs
        # Log detailed error information to help with troubleshooting
        logger.error(f"Error running server: {e}")
        logger.error("Check server configuration and system resources")
        return ERROR_EXIT_CODE


def version_command(_: argparse.Namespace) -> int:
    """
    Display version information for the FAISSx server.

    This function prints the current version of the FAISSx server to stdout.
    It's designed to be called from the CLI when the --version flag is used.

    Args:
        _: Unused argument (required for command interface consistency)
           This parameter is present to match the expected signature for CLI commands

    Returns:
        int: Always returns SUCCESS_EXIT_CODE (0) as version display cannot fail

    Example:
        >>> version_command(None)
        FAISSx Server v0.0.3
        0
    """
    # Display version information to stdout
    # Format matches common CLI tool conventions
    print(f"FAISSx Server v{__version__}")
    return SUCCESS_EXIT_CODE


def setup_run_parser(subparsers: argparse._SubParsersAction) -> None:
    """
    Configure the argument parser for the 'run' command.

    This function sets up comprehensive command-line arguments for the FAISSx server
    run command, including network configuration, authentication options, data persistence
    settings, and logging configuration. It provides sensible defaults and validation.

    Argument Categories:
        - Network: port, bind-address
        - Authentication: auth-keys, auth-file, enable-auth
        - Storage: data-dir
        - Logging: log-level

    Args:
        subparsers: ArgumentParser subparsers object to add the run command to.
                   This is typically obtained from parser.add_subparsers()

    Note:
        This function modifies the subparsers object by adding a new 'run' command
        with all its associated arguments and default values.

    Example:
        >>> parser = argparse.ArgumentParser()
        >>> subparsers = parser.add_subparsers()
        >>> setup_run_parser(subparsers)
    """
    # Create the run command parser with comprehensive help text
    # This creates a subparser specifically for the 'run' command under the main CLI
    parser = subparsers.add_parser(
        "run",
        help="Run the FAISSx server",
        description="Start the FAISSx vector database proxy server with specified configuration",
    )

    # Network configuration options
    # These arguments control how and where the server listens for connections
    # Network settings are critical for proper server operation and client connectivity

    # Port argument: determines which TCP port the ZeroMQ server binds to
    # Must be an integer in the valid port range (typically 1024-65535 for non-root)
    parser.add_argument(
        "--port",
        type=int,                           # Ensures input is converted to integer
        default=DEFAULT_PORT,               # Use predefined constant for consistency
        help=f"Port to listen on (default: {DEFAULT_PORT})",
    )

    # Bind address argument: determines which network interface to bind to
    # "0.0.0.0" means bind to all available interfaces (typical for servers)
    # Could be set to "127.0.0.1" for localhost-only or specific IP for single interface
    parser.add_argument(
        "--bind-address",
        default=DEFAULT_BIND_ADDRESS,       # Use predefined constant for consistency
        help=f"Network address to bind to (default: {DEFAULT_BIND_ADDRESS})",
    )

    # Authentication configuration options
    # These arguments control client authentication and multi-tenant access
    # Authentication is optional but recommended for production deployments

    # API keys argument: allows inline specification of key-tenant mappings
    # Format is comma-separated pairs like "api123:tenant1,xyz789:tenant2"
    # This is convenient for simple setups or testing environments
    parser.add_argument(
        "--auth-keys",
        help="API keys in format 'key1:tenant1,key2:tenant2' for authentication",
    )

    # Auth file argument: allows loading key-tenant mappings from a JSON file
    # More secure than inline keys as the file can have restricted permissions
    # Preferred for production environments with many keys or sensitive data
    parser.add_argument(
        "--auth-file",
        help='Path to JSON file containing API keys mapping (e.g., {"key1": "tenant1"})',
    )

    # Enable auth flag: boolean switch to activate authentication system
    # Must be set to True when using either auth-keys or auth-file
    # Provides explicit control over authentication activation
    parser.add_argument(
        "--enable-auth",
        action="store_true",                # Sets boolean flag when present
        help="Enable authentication (required when using auth-keys or auth-file)",
    )

    # Data persistence configuration
    # Controls where FAISS indices are stored for persistence across server restarts
    # Without this, all data is stored in memory and lost when server stops

    # Data directory argument: specifies filesystem location for index storage
    # If not provided, server operates in memory-only mode (faster but non-persistent)
    # Directory must be writable by the server process for proper operation
    parser.add_argument(
        "--data-dir",
        help="Directory to store FAISS indices for persistence (optional, uses memory if not set)",
    )

    # Logging configuration options
    # Controls the verbosity and detail level of server logging output
    # Higher verbosity levels provide more debugging information but may impact performance

    # Log level argument: determines which log messages are displayed
    # DEBUG: Most verbose, includes all internal operations
    # INFO: General information about server operations
    # WARNING: Only warnings and errors (good for production)
    # ERROR: Only error messages
    # CRITICAL: Only critical system failures
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,          # Use predefined constant for consistency
        choices=VALID_LOG_LEVELS,           # Restrict to valid options only
        help=f"Logging verbosity level (default: {DEFAULT_LOG_LEVEL})",
    )

    # Associate this parser with the run_command function
    # When the 'run' command is invoked, argparse will call run_command with parsed args
    # This is the mechanism that connects CLI parsing to actual command execution
    parser.set_defaults(func=run_command)


def main() -> int:
    """
    Main entry point for the FAISSx CLI.

    This function orchestrates the complete CLI workflow including argument parsing,
    command routing, and error handling. It serves as the primary interface between
    the user and the FAISSx server functionality.

    CLI Workflow:
        1. Create and configure the main argument parser
        2. Set up all available subcommands (run, etc.)
        3. Parse command-line arguments provided by the user
        4. Route to appropriate command handler based on user input
        5. Handle special cases (version display, help text)
        6. Return appropriate exit codes for shell scripting

    Available Commands:
        - run: Start the FAISSx server with specified configuration
        - --version: Display version information and exit

    Args:
        None (reads from sys.argv)

    Returns:
        int: Exit code for shell integration:
             - SUCCESS_EXIT_CODE (0): Successful operation
             - ERROR_EXIT_CODE (1): Error occurred (currently unused in main)

    Example:
        >>> # From command line:
        >>> # faissx-server run --port 8080 --enable-auth
        >>> # faissx-server --version
        >>> exit_code = main()
    """
    # Create main argument parser with comprehensive description and help formatting
    # This is the top-level parser that handles global options and command routing
    parser = argparse.ArgumentParser(
        description="FAISSx Server - A high-performance vector database proxy",
        epilog="For more information about specific commands, use: faissx-server <command> --help",
        formatter_class=argparse.RawDescriptionHelpFormatter,  # Preserves line breaks in help
    )

    # Add global version flag for easy access from any command level
    # This allows users to check version without needing to know subcommands
    # Global flags are processed before subcommand routing
    parser.add_argument(
        "--version",
        action="store_true",                # Sets boolean flag when present
        help="Show version information and exit"
    )

    # Set up command subparsers for different CLI operations
    # This creates an extensible command structure (run, status, backup, etc.)
    # Each subcommand can have its own specific arguments and help text
    subparsers = parser.add_subparsers(
        dest="command",                     # Stores selected command in args.command
        help="Available commands (use <command> --help for details)",
        metavar="COMMAND",                  # Shows "COMMAND" instead of command list in usage
    )

    # Configure all available subcommands by calling their setup functions
    # This modular approach makes it easy to add new commands in the future
    # Each command's setup function adds its specific arguments and handlers
    setup_run_parser(subparsers)

    # Parse command-line arguments provided by the user via sys.argv
    # This processes all arguments and validates them according to our definitions
    # Returns a Namespace object containing all parsed argument values
    args = parser.parse_args()

    # Handle global version flag at top level (takes precedence over subcommands)
    # This check happens before any subcommand processing
    # Allows "faissx-server --version" to work regardless of other arguments
    if args.version:
        return version_command(args)

    # Route to appropriate command handler based on parsed arguments
    # The 'func' attribute is set by parser.set_defaults() in each subcommand setup
    if hasattr(args, "func"):
        # A valid command was specified - execute its associated function
        # The function receives the full args namespace with all parsed values
        return args.func(args)
    else:
        # No command specified - show help and exit successfully
        # This is not an error condition, just guidance for the user
        # Helps users understand available commands when they run "faissx-server" alone
        parser.print_help()
        return SUCCESS_EXIT_CODE


if __name__ == "__main__":
    sys.exit(main())
