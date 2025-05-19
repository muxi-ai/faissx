#!/usr/bin/env python33
"""
FAISSx Server Runner

This script runs the FAISSx server using environment variables for configuration,
making it suitable for Docker deployments.
"""

import os
from faissx import server


if __name__ == "__main__":
    # Read configuration from environment variables
    port = int(os.environ.get("FAISSX_PORT", "45678"))
    bind_address = os.environ.get("FAISSX_BIND_ADDRESS", "0.0.0.0")
    data_dir = os.environ.get("FAISSX_DATA_DIR")
    enable_auth = os.environ.get("FAISSX_ENABLE_AUTH", "").lower() in [
        "true",
        "1",
        "yes",
    ]

    # Handle authentication - use either auth_keys or auth_file
    auth_keys = None
    auth_file = os.environ.get("FAISSX_AUTH_FILE")

    # Parse API keys from environment if provided
    if os.environ.get("FAISSX_AUTH_KEYS"):
        auth_keys = {}
        try:
            for key_pair in os.environ.get("FAISSX_AUTH_KEYS").split(","):
                api_key, tenant_id = key_pair.strip().split(":")
                auth_keys[api_key] = tenant_id
        except Exception as e:
            print(f"Error parsing API keys from environment: {e}")

    # Warning for auth enabled but no keys
    if enable_auth and not auth_keys and not auth_file:
        print("Warning: Authentication enabled but no keys provided")

    try:
        # Configure server
        server.configure(
            port=port,
            bind_address=bind_address,
            auth_keys=auth_keys,
            auth_file=auth_file,
            enable_auth=enable_auth,
            data_dir=data_dir,  # Use None if not specified
        )

        # Run server
        server.run()
    except ValueError as e:
        print(f"Error configuring server: {e}")
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error running server: {e}")
