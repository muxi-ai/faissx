#!/usr/bin/env python3
"""
FAISSx Error Recovery and Reconnection Example

This example demonstrates how to use the error recovery and reconnection capabilities in FAISSx:
1. Configure error recovery with custom retry and backoff settings
2. Register callbacks for disconnection and reconnection events
3. Use the retry mechanism for operations
4. Handle server disconnections gracefully

These features make client applications more resilient to network issues and server restarts,
providing a more robust experience in production environments.
"""

import time
import numpy as np
import signal
import sys
import os

# Import FAISSx client
from faissx import client as faiss
from faissx.client.recovery import (
    configure_recovery,
    on_reconnect,
    on_disconnect,
    with_retry,
    is_connected,
    force_reconnect
)

# Set up logging to see what's happening
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DIMENSION = 64
NUM_VECTORS = 1000
SERVER_URL = os.environ.get("FAISSX_SERVER", "tcp://localhost:45678")
API_KEY = os.environ.get("FAISSX_API_KEY", "test-key-1")
TENANT_ID = os.environ.get("FAISSX_TENANT_ID", "tenant-1")

# Server control (simulating a server restart)
server_process = None


def generate_random_vectors(count, dimension):
    """Generate random vectors for testing."""
    return np.random.random((count, dimension)).astype('float32')


def reconnect_callback():
    """Called when the client successfully reconnects to the server."""
    logger.info("🔄 Reconnected to server! Resuming operations...")


def disconnect_callback():
    """Called when the client detects a disconnection from the server."""
    logger.info("❌ Disconnected from server! Will attempt to reconnect...")


def setup_client():
    """Configure the FAISSx client with recovery settings."""
    # Configure the client for remote operation
    faiss.configure(
        server=SERVER_URL,
        api_key=API_KEY,
        tenant_id=TENANT_ID
    )

    # Configure recovery settings
    configure_recovery(
        max_retries=5,             # Maximum number of retry attempts
        initial_backoff=0.5,       # Start with a short backoff (500ms)
        max_backoff=10.0,          # Maximum backoff time (10 seconds)
        backoff_factor=1.5,        # Exponential growth factor
        jitter=0.2,                # Add random jitter to avoid thundering herd
        enabled=True,              # Enable recovery mechanism
        auto_reconnect=True        # Automatically attempt reconnection
    )

    # Register callbacks
    on_reconnect(reconnect_callback)
    on_disconnect(disconnect_callback)


def perform_operations_with_retry():
    """Demonstrate operations with automatic retry."""
    db_vectors = generate_random_vectors(NUM_VECTORS, DIMENSION)
    query_vectors = generate_random_vectors(5, DIMENSION)

    # Create an index
    logger.info("Creating index...")
    index = faiss.IndexFlatL2(DIMENSION)

    # Add vectors with retry
    logger.info("Adding vectors with retry mechanism...")
    try:
        # The with_retry function will automatically retry on connection errors
        with_retry(index.add, db_vectors)
        logger.info(f"Successfully added {index.ntotal} vectors")
    except Exception as e:
        logger.error(f"Failed to add vectors after retries: {e}")

    # Search with retry
    logger.info("Performing search with retry mechanism...")
    try:
        distances, indices = with_retry(index.search, query_vectors, k=5)
        logger.info(f"Search successful: found {len(indices)} results")
    except Exception as e:
        logger.error(f"Failed to perform search after retries: {e}")


def simulate_server_restart():
    """Simulate a server restart or network disruption."""
    logger.info("⚠️ Simulating server restart or network disruption...")

    # This would be done by stopping and starting a real server
    # For this example, we'll just introduce a delay
    time.sleep(5)

    logger.info("Server is back online!")


def continuous_operation():
    """Demonstrate continuous operation with automatic reconnection."""
    logger.info("Starting continuous operation...")

    # Create an index
    index = faiss.IndexFlatL2(DIMENSION)

    # Run continuous operations
    try:
        for i in range(30):
            try:
                # Generate small batches
                vectors = generate_random_vectors(10, DIMENSION)

                # Add vectors - will be retried automatically if server is down
                with_retry(index.add, vectors)
                logger.info(f"Added batch {i+1}, total vectors: {index.ntotal}")

                # Check connection status
                if i == 10:
                    # Simulate server disruption after 10 batches
                    simulate_server_restart()

                # Small delay between operations
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Operation failed: {e}")

    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")


def force_reconnection_demo():
    """Demonstrate how to force a reconnection."""
    logger.info("Testing forced reconnection...")

    # Create an index and add vectors
    index = faiss.IndexFlatL2(DIMENSION)
    with_retry(index.add, generate_random_vectors(100, DIMENSION))

    # Check connection status
    logger.info(f"Connected: {is_connected()}")

    # Force reconnection even if we're still connected
    logger.info("Forcing reconnection...")
    force_reconnect()

    # Wait for reconnection
    time.sleep(2)

    # Check if we can still operate
    try:
        count = with_retry(lambda: index.ntotal)
        logger.info(f"Reconnection successful, index has {count} vectors")
    except Exception as e:
        logger.error(f"Failed to reconnect: {e}")


def handle_exit(*args):
    """Handle program exit gracefully."""
    logger.info("Exiting...")
    sys.exit(0)


def main():
    """Main function demonstrating error recovery features."""
    # Set up signal handling
    signal.signal(signal.SIGINT, handle_exit)

    logger.info("FAISSx Error Recovery Example")
    logger.info("============================")

    # Setup client with recovery options
    setup_client()

    # Basic operations with retry
    perform_operations_with_retry()

    # Force reconnection demo
    force_reconnection_demo()

    # Run continuous operations with a simulated server disruption
    continuous_operation()


if __name__ == "__main__":
    main()
