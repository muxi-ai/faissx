#!/usr/bin/env python3
"""
FAISS Proxy Server CLI

Command-line interface for the FAISS Proxy server.
"""

import sys
import argparse
from faiss_proxy import server
from faiss_proxy import __version__


def run_command(args):
    """Run the server with the specified arguments"""
    # Parse API keys if provided
    auth_keys = None
    if args.api_keys:
        auth_keys = {}
        try:
            for key_pair in args.api_keys.split(","):
                api_key, tenant_id = key_pair.strip().split(":")
                auth_keys[api_key] = tenant_id
        except Exception as e:
            print(f"Error parsing API keys: {e}")
            return 1

    # If both auth methods are provided, show an error
    if args.api_keys and args.auth_file:
        print("Error: Cannot provide both --auth-keys and --auth-file")
        return 1

    # Configure server
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

    print(f"Starting FAISS Proxy Server on {args.bind_address}:{args.port}")
    if args.data_dir:
        print(f"Data directory: {args.data_dir}")
    else:
        print("Using in-memory indices (no persistence)")
    print(f"Authentication enabled: {args.enable_auth}")
    if args.auth_file:
        print(f"Loading authentication keys from: {args.auth_file}")

    # Run server
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
    """Show version information"""
    print(f"FAISS Proxy Server v{__version__}")
    return 0


def setup_run_parser(subparsers):
    """Set up the 'run' command parser"""
    parser = subparsers.add_parser(
        "run", help="Run the FAISS Proxy server"
    )
    parser.add_argument("--port", type=int, default=45678, help="Port to listen on")
    parser.add_argument("--bind-address", default="0.0.0.0", help="Address to bind to")
    parser.add_argument("--auth-keys", help="API keys in format key1:tenant1,key2:tenant2")
    parser.add_argument(
        "--auth-file",
        help="Path to JSON file containing API keys mapping (e.g., {\"key1\": \"tenant1\"})"
    )
    parser.add_argument("--enable-auth", action="store_true", help="Enable authentication")
    parser.add_argument("--data-dir", help="Directory to store FAISS indices (optional)")
    parser.set_defaults(func=run_command)


def main():
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(
        description="FAISS Proxy Server - A high-performance vector database proxy"
    )
    parser.add_argument(
        "--version", action="store_true", help="Show version information"
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Set up command parsers
    setup_run_parser(subparsers)

    # Parse arguments
    args = parser.parse_args()

    # Handle --version at the top level
    if args.version:
        return version_command(args)

    # Execute command if provided
    if hasattr(args, "func"):
        return args.func(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
