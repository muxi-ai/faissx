#!/usr/bin/env python
"""
FAISSx Combined Optimization and Recovery Example

This example demonstrates how to use both optimization controls and error recovery
capabilities together for a robust and high-performance implementation.

Key features demonstrated:
1. Setting up error recovery with exponential backoff
2. Configuring fine-grained performance parameters
3. Memory management for efficient resource usage
4. Handling network disruptions gracefully
"""

import time
import numpy as np
import logging
import os
import random

# Import FAISSx client
from faissx import client as faiss

# Import optimization and recovery utilities
from faissx.client.recovery import (
    configure_recovery,
    on_reconnect,
    on_disconnect,
    with_retry
)
from faissx.client.optimization import memory_manager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DIMENSION = 128
NUM_VECTORS = 10000
SERVER_URL = os.environ.get("FAISSX_SERVER", "tcp://localhost:45678")
API_KEY = os.environ.get("FAISSX_API_KEY", "test-key-1")
TENANT_ID = os.environ.get("FAISSX_TENANT_ID", "tenant-1")


def setup_client():
    """Configure the client with both recovery and optimization settings."""
    logger.info("Configuring FAISSx client...")

    # 1. Configure the client for remote execution
    faiss.configure(
        server=SERVER_URL,
        api_key=API_KEY,
        tenant_id=TENANT_ID
    )

    # 2. Configure error recovery settings
    configure_recovery(
        max_retries=3,              # Number of retry attempts
        initial_backoff=0.5,        # Start with 500ms backoff
        max_backoff=10.0,           # Maximum backoff of 10 seconds
        backoff_factor=2.0,         # Double backoff time each attempt
        jitter=0.1,                 # Add 10% random jitter
        enabled=True,               # Enable recovery
        auto_reconnect=True         # Automatically reconnect on failure
    )

    # 3. Register callbacks for connection events
    on_reconnect(lambda: logger.info("🔄 Reconnected to server!"))
    on_disconnect(lambda: logger.info("❌ Connection lost to server!"))

    # 4. Configure memory management settings
    memory_manager.set_option('max_memory_usage_mb', 1024)   # 1GB limit
    memory_manager.set_option('use_memory_mapping', True)    # Use memory mapping for large indices
    memory_manager.set_option('index_cache_size', 50)        # Cache up to 50 indices
    memory_manager.set_option('auto_unload_unused_indices', True)  # Auto-unload unused indices

    logger.info("Client configured with optimization and recovery settings")


def generate_vectors(count, dimension):
    """Generate random vectors for testing."""
    return np.random.random((count, dimension)).astype('float32')


def create_optimized_index():
    """Create an IVF index with optimized parameters."""
    logger.info("Creating optimized IVF index...")

    # Create vectors
    vectors = generate_vectors(NUM_VECTORS, DIMENSION)

    # Create an IVF index with 100 clusters
    nlist = 100
    index = faiss.IndexIVFFlat(DIMENSION, nlist)

    # Set optimization parameters
    index.set_parameter('nprobe', 10)  # Search 10 clusters (balance between speed and accuracy)
    index.set_parameter('n_iter', 20)  # Training iterations
    index.set_parameter('batch_size', 5000)  # Process in batches of 5000 vectors

    # Train and add vectors with automatic retry
    logger.info("Training index...")
    with_retry(index.train, vectors)

    logger.info("Adding vectors...")
    with_retry(index.add, vectors)

    logger.info(f"Created index with {index.ntotal} vectors")
    return index


def perform_search_operations(index, query_count=10):
    """Perform search operations with error recovery."""
    logger.info(f"Performing {query_count} search operations...")

    for i in range(query_count):
        try:
            # Generate a query vector
            query = generate_vectors(1, DIMENSION)

            # Perform search with automatic retry
            start_time = time.time()
            distances, indices = with_retry(index.search, query, k=5)
            elapsed = time.time() - start_time

            logger.info(f"Query {i+1}: found {len(indices[0])} results in {elapsed:.4f}s")

            # Simulate random delay between queries
            time.sleep(random.uniform(0.1, 0.5))

            # Simulate random network issue (1 in 10 chance)
            if random.random() < 0.1:
                logger.warning("Simulating temporary network issue...")
                time.sleep(2)  # Pause to simulate disruption

        except Exception as e:
            logger.error(f"Error during search: {e}")


def main():
    """Main function demonstrating combined optimization and recovery features."""
    logger.info("FAISSx Combined Optimization and Recovery Example")
    logger.info("===============================================")

    # Setup client with both optimization and recovery
    setup_client()

    try:
        # Create optimized index with performance parameters
        index = create_optimized_index()

        # Perform operations with automatic error recovery
        perform_search_operations(index)

        # Check optimization statistics
        params = index.get_parameters()
        logger.info("Index parameters:")
        for name, value in params.items():
            logger.info(f"  {name}: {value}")

        # Check memory usage
        size_bytes = index.estimate_memory_usage()
        size_mb = size_bytes / (1024 * 1024)
        logger.info(f"Estimated memory usage: {size_mb:.2f} MB")

    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")

    logger.info("Example completed")


if __name__ == "__main__":
    main()
