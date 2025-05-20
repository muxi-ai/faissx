#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Force remote mode with no fallbacks for FAISSx
#
# Copyright (C) 2025 Ran Aroussi

"""
Force the FAISSx client to use remote mode and fail on errors instead of falling back.

This module patches the FAISSx client to enforce remote mode connections and
disable fallback to local mode when remote operations fail.
"""

import os
import sys
import logging
import time
import zmq
import msgpack
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path to import faissx
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(current_dir), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import faissx
import faissx
from faissx.client.client import FaissXClient


def test_server_connection(server_url):
    """
    Test direct connection to ZMQ server.

    Args:
        server_url: The URL of the ZMQ server

    Returns:
        bool: True if connection successful, False otherwise
    """
    # Initialize ZMQ context and socket
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, 3000)  # 3 seconds timeout

    try:
        # Connect to server
        socket.connect(server_url)

        # Create a ping message
        ping_message = {"action": "ping"}

        # Serialize with msgpack
        packed_msg = msgpack.packb(ping_message, use_bin_type=True)

        # Send message
        socket.send(packed_msg)

        # Receive response
        response = socket.recv()
        unpacked = msgpack.unpackb(response, raw=False)

        logger.info(f"Direct ZMQ connection test successful: {unpacked}")
        return True
    except Exception as e:
        logger.warning(f"Direct ZMQ connection test failed: {e}")
        return False
    finally:
        # Clean up
        socket.close()
        context.term()


def force_remote_mode(server_url: str = "tcp://localhost:45679", max_retries: int = 5):
    """
    Force the FAISSx client to use remote mode with no fallback to local mode.

    Args:
        server_url: The URL of the remote server to connect to
        max_retries: Maximum number of connection retry attempts before failing

    This function applies several patches to enforce remote mode:
    1. Sets the FAISSX_FALLBACK_TO_LOCAL environment variable to "0"
    2. Sets the _FALLBACK_TO_LOCAL module variable to False
    3. Monkeypatches the get_client function to retry connections and fail instead of falling back
    """
    # Ensure fallback is disabled via environment variable
    os.environ["FAISSX_FALLBACK_TO_LOCAL"] = "0"

    # First, test direct connection to server
    logger.info(f"Testing direct connection to server at {server_url}...")
    if test_server_connection(server_url):
        logger.info("Direct connection test successful")
    else:
        logger.warning("Direct connection test failed")

    # Disable fallback in the module
    faissx.client._FALLBACK_TO_LOCAL = False

    # Configure the client with remote server
    faissx.configure(url=server_url)
    logger.info(f"FAISSx configured to connect to {server_url} with no fallbacks")

    # Override the get_client function to retry connections
    def retry_get_client() -> Optional[FaissXClient]:
        """
        Get a client with retry logic and no fallbacks.

        Returns:
            The configured FaissXClient

        Raises:
            RuntimeError: If connection fails after retries
        """
        # Clear any existing client instances
        faissx.client.client._client = None

        # Check if server URL is empty
        if not faissx._API_URL or faissx._API_URL == "":
            logger.info("No server URL configured, returning None for local mode")
            return None

        # Retry connection attempts
        retry_count = 0
        last_error = None

        while retry_count <= max_retries:
            try:
                # Attempt to create a client instance
                client = FaissXClient()
                logger.info("Successfully connected to FAISSx server")
                return client
            except Exception as e:
                last_error = e
                retry_count += 1
                logger.warning(
                    f"Failed to connect to FAISSx server (attempt {retry_count}/{max_retries+1}): {e}"
                )

                # If we haven't reached max retries, wait before trying again
                if retry_count <= max_retries:
                    time.sleep(1)  # Wait 1 second between retry attempts

        # All retries failed, raise error
        raise RuntimeError(f"Failed to connect to FAISSx server after {max_retries+1} attempts: {last_error}")

    # Apply the monkeypatch
    from faissx.client import client as client_module
    client_module.get_client = retry_get_client
    logger.info("Patched get_client function to enforce retries and no fallbacks")


def main():
    """Run a simple test to verify forced remote mode is working."""
    # Force remote mode
    force_remote_mode()

    # Try to create an index
    try:
        index = faissx.client.IndexFlatL2(32)
        print(f"Successfully created index: {index}")

        # Test adding and searching
        vectors = [[1.0, 2.0] * 16, [3.0, 4.0] * 16]
        index.add(vectors)
        print(f"Added {len(vectors)} vectors, total: {index.ntotal}")

        print("Remote mode is working correctly!")

    except Exception as e:
        print(f"Error using remote mode: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
