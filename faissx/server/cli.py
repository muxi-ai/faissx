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

Command-line interface for the FAISSx server. This module provides a CLI
for running and managing the FAISSx vector database proxy server, including
configuration of authentication, data persistence, and network settings.
"""

import sys
import argparse
from faissx import server
from faissx import __version__


def run_command(args):
    """
    Run the FAISSx server with the specified command-line arguments.

    This function handles server configuration and startup, including:
    - Parsing and validating API keys
    - Configuring server settings
    - Starting the server process
    - Handling graceful shutdown

    Args:
        args: Command-line arguments parsed by argparse

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    # Parse API keys from command line if provided
    auth_keys = None
    if args.auth_keys:
        auth_keys = {}
        try:
            # Split comma-separated key:tenant pairs and store in dictionary
            for key_pair in args.auth_keys.split(","):
                api_key, tenant_id = key_pair.strip().split(":")
                auth_keys[api_key] = tenant_id
        except Exception as e:
            print(f"Error parsing API keys: {e}")
            return 1

    # Validate that only one authentication method is specified
    if args.auth_keys and args.auth_file:
        print("Error: Cannot provide both --auth-keys and --auth-file")
        return 1

    # Configure server with provided settings
    try:
        server.configure(
            port=args.port,
            bind_address=args.bind_address,
            auth_keys=auth_keys,
            auth_file=args.auth_file,
            enable_auth=args.enable_auth,
            data_dir=args.data_dir,
        )
    except ValueError as e:
        print(f"Error configuring server: {e}")
        return 1

    # Print server startup information
    print(f"Starting FAISSx Server on {args.bind_address}:{args.port}")
    if args.data_dir:
        print(f"Data directory: {args.data_dir}")
    else:
        print("Using in-memory indices (no persistence)")
    print(f"Authentication enabled: {args.enable_auth}")
    if args.auth_file:
        print(f"Loading authentication keys from: {args.auth_file}")

    # Start server and handle shutdown
    try:
        server.run()
        return 0
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        return 0
    except Exception as e:
        print(f"Error running server: {e}")
        return 1


def version_command(_):
    """
    Display version information for the FAISSx server.

    Args:
        _: Unused argument (required for command interface)

    Returns:
        int: Exit code (0 for success)
    """
    print(f"FAISSx Server v{__version__}")
    return 0


def setup_run_parser(subparsers):
    """
    Configure the argument parser for the 'run' command.

    This function sets up all command-line arguments specific to running
    the FAISSx server, including network settings, authentication options,
    and data persistence configuration.

    Args:
        subparsers: ArgumentParser subparsers object to add the run command to
    """
    parser = subparsers.add_parser(
        "run", help="Run the FAISSx server"
    )
    # Network configuration
    parser.add_argument("--port", type=int, default=45678, help="Port to listen on")
    parser.add_argument("--bind-address", default="0.0.0.0", help="Address to bind to")

    # Authentication configuration
    parser.add_argument("--auth-keys", help="API keys in format key1:tenant1,key2:tenant2")
    parser.add_argument(
        "--auth-file",
        help="Path to JSON file containing API keys mapping (e.g., {\"key1\": \"tenant1\"})"
    )
    parser.add_argument("--enable-auth", action="store_true", help="Enable authentication")

    # Data persistence
    parser.add_argument("--data-dir", help="Directory to store FAISS indices (optional)")

    # Set the function to call when this command is used
    parser.set_defaults(func=run_command)


def main():
    """
    Main entry point for the FAISSx CLI.

    This function:
    1. Sets up the argument parser with all available commands
    2. Parses command-line arguments
    3. Executes the appropriate command based on user input
    4. Handles version display and help text

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    # Create main argument parser
    parser = argparse.ArgumentParser(
        description="FAISSx Server - A high-performance vector database proxy"
    )
    parser.add_argument(
        "--version", action="store_true", help="Show version information"
    )

    # Set up command subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    setup_run_parser(subparsers)

    # Parse arguments and execute command
    args = parser.parse_args()

    # Handle version flag at top level
    if args.version:
        return version_command(args)

    # Execute command if specified, otherwise show help
    if hasattr(args, "func"):
        return args.func(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
