#!/usr/bin/env python3
"""
FAISSx Server Runner

This script runs the FAISSx server using environment variables for configuration,
making it suitable for Docker deployments.
"""

import os
import logging
from faissx import server
from faissx.server.server import (
    DEFAULT_SOCKET_TIMEOUT,
    DEFAULT_HIGH_WATER_MARK,
    DEFAULT_LINGER
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("faissx.runner")


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

    # Socket configuration
    socket_timeout = int(os.environ.get("FAISSX_SOCKET_TIMEOUT", str(DEFAULT_SOCKET_TIMEOUT)))
    high_water_mark = int(os.environ.get("FAISSX_HIGH_WATER_MARK", str(DEFAULT_HIGH_WATER_MARK)))
    linger = int(os.environ.get("FAISSX_LINGER", str(DEFAULT_LINGER)))

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
            logger.error(f"Error parsing API keys from environment: {e}")

    # Warning for auth enabled but no keys
    if enable_auth and not auth_keys and not auth_file:
        logger.warning("Authentication enabled but no keys provided")

    try:
        # Configure server
        server.configure(
            port=port,
            bind_address=bind_address,
            auth_keys=auth_keys,
            auth_file=auth_file,
            enable_auth=enable_auth,
            data_dir=data_dir,  # Use None if not specified
            socket_timeout=socket_timeout,
            high_water_mark=high_water_mark,
            linger=linger
        )

        # Run server
        server.run()
    except ValueError as e:
        logger.error(f"Error configuring server: {e}")
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.exception(f"Error running server: {e}")
