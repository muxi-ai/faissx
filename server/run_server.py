#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FAISSx Server Runner Script
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
FAISSx Server Runner Script

This script provides a convenient way to start the FAISSx server
with command-line arguments as described in the README.md.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path

from faissx.server.server import run_server


def parse_auth_keys(auth_keys_str):
    """Parse auth keys string in format 'key1:tenant1,key2:tenant2'."""
    if not auth_keys_str:
        return {}

    auth_keys = {}
    pairs = auth_keys_str.split(',')
    for pair in pairs:
        if ':' in pair:
            key, tenant = pair.split(':', 1)
            auth_keys[key.strip()] = tenant.strip()

    return auth_keys


def load_auth_file(auth_file):
    """Load auth keys from a JSON file."""
    if not auth_file or not os.path.exists(auth_file):
        return {}

    try:
        with open(auth_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading auth file: {e}")
        return {}


def main():
    """Parse command-line arguments and start the server."""
    parser = argparse.ArgumentParser(description="FAISSx Server")

    # Basic configuration
    parser.add_argument("--port", type=int, default=45678,
                        help="Port to listen on (default: 45678)")
    parser.add_argument("--bind-address", default="0.0.0.0",
                        help="Address to bind to (default: 0.0.0.0)")
    parser.add_argument("--data-dir",
                        help="Directory for persistent storage (None = in-memory only)")

    # Authentication
    parser.add_argument("--enable-auth", action="store_true",
                        help="Enable authentication")
    parser.add_argument("--auth-keys",
                        help="API keys in format 'key1:tenant1,key2:tenant2'")
    parser.add_argument("--auth-file",
                        help="Path to JSON file containing API keys")

    # Logging
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--log-file",
                        help="Log to a file instead of stdout")

    # Parse arguments
    args = parser.parse_args()

    # Configure logging
    log_level = getattr(logging, args.log_level)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    if args.log_file:
        logging.basicConfig(filename=args.log_file, level=log_level, format=log_format)
    else:
        logging.basicConfig(level=log_level, format=log_format)

    # Use environment variables as fallback, but prioritize command-line arguments
    port = int(os.environ.get("FAISSX_PORT", args.port))
    bind_address = os.environ.get("FAISSX_BIND_ADDRESS", args.bind_address)
    data_dir = args.data_dir or os.environ.get("FAISSX_DATA_DIR")
    enable_auth = args.enable_auth or os.environ.get("FAISSX_ENABLE_AUTH") == "1"

    # Parse auth keys
    auth_keys = {}
    if args.auth_keys:
        auth_keys = parse_auth_keys(args.auth_keys)
    elif args.auth_file:
        auth_keys = load_auth_file(args.auth_file)
    elif os.environ.get("FAISSX_AUTH_FILE"):
        auth_keys = load_auth_file(os.environ.get("FAISSX_AUTH_FILE"))

    # Convert data_dir to Path if provided
    if data_dir:
        data_dir = Path(data_dir)

    # Print startup information
    print(f"Starting FAISSx server on {bind_address}:{port}")
    print(f"Data directory: {data_dir or 'None (in-memory mode)'}")
    print(f"Authentication: {'Enabled' if enable_auth else 'Disabled'}")

    if enable_auth:
        print(f"Auth keys: {len(auth_keys)} key(s) configured")

    # Run the server
    try:
        run_server(
            port=port,
            bind_address=bind_address,
            auth_keys=auth_keys,
            enable_auth=enable_auth,
            data_dir=data_dir
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
