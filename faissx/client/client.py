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
FAISSx Client Module for Vector Similarity Search.

This module provides a client interface for the FAISSx vector similarity search service.
It supports both local and remote modes of operation, with the remote mode connecting
to a FAISSx server via ZeroMQ messaging.

Key components:
- FaissXClient: Main client class that handles communication with the server
- Utility functions for retry logic and timeouts
- Type aliases and constants for configuration
- Global singleton client instance and configuration functions

Architecture:
The client operates in two distinct modes:
1. Local Mode (Default): Uses local FAISS library directly with no server connection
2. Remote Mode: Connects to a FAISSx server via ZeroMQ for distributed operation

Communication Protocol:
- Uses ZeroMQ (zmq) for efficient binary messaging
- Serializes data using msgpack for compact representation
- Implements retry logic for handling transient failures
- Supports authentication via API keys and tenant isolation

Typical usage:
    # Configure the client for remote operation
    from faissx.client import configure
    configure(server="tcp://localhost:5555", api_key="your-key", tenant_id="tenant1")

    # Create and use an index
    from faissx.client.indices import IndexFlatL2
    index = IndexFlatL2(128)
    index.add(vectors)
    distances, indices = index.search(query, k=5)

    # Or use local mode (default) without configuration
    from faissx.client.indices import IndexFlatL2
    index = IndexFlatL2(128)  # Uses local FAISS under the hood

Environment Variables:
- FAISSX_SERVER: ZeroMQ server address (e.g., "tcp://localhost:45678")
- FAISSX_API_KEY: API key for authentication
- FAISSX_TENANT_ID: Tenant identifier for multi-tenant isolation

Error Handling:
The client implements robust error handling with:
- Automatic retries with exponential backoff
- Timeout management for long-running operations
- Detailed error messages with context information
"""

import os
import zmq
import msgpack
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Union, Callable
import time
from functools import wraps

# Import from our custom timeout module
from .timeout import timeout as operation_timeout, TimeoutError, set_timeout

# Configure logging for the module
logger = logging.getLogger(__name__)

# Default values for parameters
DEFAULT_TIMEOUT = 5.0              # Default timeout in seconds for server communication
DEFAULT_MAX_RETRIES = 2            # Default number of retry attempts for operations
DEFAULT_RETRY_DELAY = 1.0          # Default delay between retries in seconds
DEFAULT_BATCH_SIZE_ADD = 1000      # Default batch size when adding vectors
DEFAULT_BATCH_SIZE_SEARCH = 100    # Default batch size when searching
DEFAULT_K = 10                     # Default number of nearest neighbors to return

# Type aliases for improved readability
# Return type for search operations - dictionary with results, status, etc.
SearchResult = Dict[str, Any]
# Type for index identifiers - unique string IDs for each index
IndexID = str
# Input vector types that can be processed - numpy arrays or lists of floats
VectorData = Union[np.ndarray, List[float]]


def retry_on_failure(max_retries: int = DEFAULT_MAX_RETRIES,
                     delay: float = DEFAULT_RETRY_DELAY) -> Callable:
    """Decorator that retries a function call on failure with exponential backoff.

    This decorator will retry the wrapped function if it raises an exception, up to
    the specified maximum number of retries, with a configurable delay between attempts.
    The delay increases exponentially with each retry attempt, providing a robust
    mechanism for handling transient failures like network issues or server overload.

    Args:
        max_retries: Maximum number of retry attempts (default: 2)
        delay: Initial delay in seconds between retries (default: 1.0)
            Each subsequent retry will wait delay * (retry_count) seconds

    Returns:
        Decorated function that will retry on failure

    Raises:
        RuntimeError: When all retry attempts fail, with details about the last error

    Example:
        @retry_on_failure(max_retries=3, delay=2.0)
        def connect_to_server():
            # Function that might fail due to network issues
            return establish_connection()

        # This will try up to 4 times (original + 3 retries),
        # with delays of 2s, 4s, and 6s between attempts
    """
    def decorator(func: Callable) -> Callable:
        """Inner decorator that wraps the target function."""
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            """Wrapper that implements the retry logic."""
            retry_count = 0
            last_error = None

            # Attempt the operation up to max_retries + 1 times
            while retry_count <= max_retries:
                try:
                    # Try calling the original function
                    return func(*args, **kwargs)
                except Exception as e:
                    # Handle failure and prepare for potential retry
                    last_error = e
                    retry_count += 1
                    logger.warning(
                        f"Operation failed (attempt {retry_count}/{max_retries+1}): {e}"
                    )
                    # Wait before retrying if we haven't exceeded max retries
                    if retry_count <= max_retries:
                        time.sleep(delay * retry_count)

            # If we've exhausted all retries, raise the last error
            raise RuntimeError(
                f"Operation failed after {max_retries+1} attempts: {last_error}"
            )

        return wrapper

    return decorator


class FaissXClient:
    """Client for interacting with FAISSx server via ZeroMQ.

    This client provides a robust interface for vector database operations including
    creating indices, adding vectors, and performing similarity searches. It abstracts
    away the complexities of client-server communication, serialization, and error handling.

    The client operates in two distinct modes:
    1. Local mode: When no server is configured, operations are performed locally using FAISS
    2. Remote mode: When connected to a FAISSx server, operations are sent to the server
       and executed there

    Features:
    - Automatic connection management and reconnection on failure
    - Request serialization and response deserialization
    - Timeout handling for long-running operations
    - Authentication with API keys
    - Multi-tenant isolation via tenant IDs
    - Batched operations for large vector sets
    - Comprehensive error handling and diagnostics

    Attributes:
        server: ZeroMQ server address (e.g., "tcp://localhost:5555")
        api_key: API key for authentication with the server
        tenant_id: Tenant identifier for multi-tenant environments
        context: ZeroMQ context object
        socket: ZeroMQ socket for communication
        mode: Operational mode ("local" or "remote")
        timeout: Connection timeout in seconds

    Example:
        ```python
        client = FaissXClient()
        client.configure(server="tcp://localhost:5555",
                         api_key="your-key",
                         tenant_id="tenant1")

        # Create an index
        index_id = client.create_index("my_index", dimension=128)

        # Add vectors
        vectors = np.random.random((100, 128)).astype('float32')
        client.add_vectors(index_id, vectors)

        # Search
        query = np.random.random((1, 128)).astype('float32')
        results = client.search(index_id, query, k=5)
        ```
    """

    def __init__(self) -> None:
        """Initialize the client with configuration from environment variables.

        This constructor sets up the client with default settings and attempts
        to load configuration from environment variables. No connection is
        established until the configure() method is called or an operation
        is performed requiring server communication.

        Environment variables:
            FAISSX_SERVER: ZeroMQ server address
            FAISSX_API_KEY: API key for authentication
            FAISSX_TENANT_ID: Tenant identifier
        """
        # Load configuration from environment variables
        self.server: str = os.environ.get("FAISSX_SERVER", "")
        self.api_key: str = os.environ.get("FAISSX_API_KEY", "")
        self.tenant_id: str = os.environ.get("FAISSX_TENANT_ID", "")

        # Initialize connection-related attributes
        self.context: Optional[zmq.Context] = None
        self.socket: Optional[zmq.Socket] = None
        self.mode: str = "local"  # Start in local mode until configured
        self.timeout: float = DEFAULT_TIMEOUT

    def configure(
        self,
        server: Optional[str] = None,
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Configure the client with server details and authentication.

        This method updates the client configuration and attempts to establish
        a connection to the server. If successful, the client switches to remote mode.
        If no server is provided and none was set previously, the client remains
        in local mode.

        Args:
            server: ZeroMQ server address
            api_key: API key for authentication
            tenant_id: Tenant identifier for multi-tenant setups
            timeout: Connection timeout in seconds (default: 5.0)

        Raises:
            RuntimeError: If unable to establish a connection to the server
        """
        # Update timeout settings
        self.timeout = timeout
        set_timeout(timeout)  # Configure global timeout in the timeout module

        # Update configuration with provided values or keep existing ones
        if server:
            self.server = server
        if api_key:
            self.api_key = api_key
        if tenant_id:
            self.tenant_id = tenant_id

        # Verify we have a server address
        if not self.server:
            logger.warning("No server address provided")
            return

        # Establish connection and set mode
        try:
            self.connect()  # Will retry based on retry_on_failure decorator
            self.mode = "remote"  # Switch to remote mode on successful connection
            logger.info(f"Successfully configured client for server {self.server}")
        except Exception as e:
            logger.error(f"Failed to configure client: {e}")
            self.mode = "local"  # Fall back to local mode on connection failure
            raise RuntimeError(f"Failed to configure client: {e}")

    def disconnect(self) -> None:
        """Close the ZeroMQ connection and clean up resources.

        This method safely releases all ZeroMQ resources, including sockets and contexts.
        It handles potential errors during cleanup gracefully to avoid exceptions
        during shutdown or garbage collection.
        """
        if self.socket:
            # Clean up socket and context
            try:
                self.socket.close()
                logger.debug("Closed ZeroMQ socket")
            except Exception as e:
                # Socket might already be closed or in an invalid state
                logger.warning(f"Error closing socket: {e}")

            if self.context:
                try:
                    self.context.term()
                    logger.debug("Terminated ZeroMQ context")
                except Exception as e:
                    # Context might already be terminated
                    logger.warning(f"Error terminating context: {e}")

            # Reset connection attributes
            self.socket = None
            self.context = None
            logger.info("Disconnected from server")

    @retry_on_failure()
    def connect(self) -> None:
        """Establish and verify a connection to the FAISSx server.

        This method handles the complete connection process:
        1. Creates a ZeroMQ context and REQ socket for request-reply pattern
        2. Configures socket options for reliable operation (timeouts, linger)
        3. Establishes the physical connection to the server
        4. Validates connectivity with a ping request
        5. Automatically retries on failure with exponential backoff

        The connection uses the ZeroMQ REQ-REP (request-reply) pattern, which
        is synchronized and ensures each request receives exactly one response.

        Socket configuration includes:
        - RCVTIMEO: Receive timeout to prevent indefinite blocking
        - LINGER: Socket close behavior for proper cleanup

        Connection validation:
        A ping request is sent after connecting to verify the server is responsive
        and properly configured to handle requests.

        Retry behavior:
        The @retry_on_failure decorator automatically retries the connection
        if it fails, with exponential backoff between attempts.

        Raises:
            ValueError: If no server address is configured
            RuntimeError: If connection fails after retries, ping fails,
                         or server returns an error response

        Note:
            This method is called automatically when:
            1. configure() is called with a server address
            2. An operation is performed that requires server communication
               when a connection doesn't already exist
        """
        # Clean up any existing connection first
        self.disconnect()

        # Initialize new ZMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)  # Request socket for req-reply pattern

        # Configure socket timeouts for reliable operation
        timeout_ms = int(self.timeout * 1000)  # Convert seconds to milliseconds
        logger.debug(f"Setting socket timeout to {self.timeout}s ({timeout_ms}ms)")
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)  # Receive timeout
        self.socket.setsockopt(zmq.LINGER, 0)  # Don't wait on close

        # Connect to server and verify connection
        if not self.server:
            raise ValueError("No server address specified")

        logger.debug(f"Connecting to {self.server}...")
        self.socket.connect(self.server)

        try:
            # Test connection with ping request
            response = self._send_request({"action": "ping"})
            if not response.get("success", False):
                # Server responded but indicated an error
                raise RuntimeError(f"Ping failed: {response.get('error', 'Unknown error')}")

            logger.info(
                f"Connected to FAISSx server at {self.server} "
                f"(response: {response})"
            )
        except Exception as e:
            # Handle connection failure - disconnect to clean up resources
            logger.error(f"Failed to connect to {self.server}: {str(e)}")
            self.disconnect()
            raise RuntimeError(f"Connection failed: {str(e)}")

    def get_client(self) -> Optional["FaissXClient"]:
        """Get an active client instance, creating one if necessary.

        Returns:
            Active client instance or None if no server is configured
        """
        if not self.socket and not self.server:
            return None
        if not self.socket and self.server:
            try:
                self.connect()
                return self
            except Exception as e:
                logger.error(f"Failed to connect: {e}")
                return None
        return self

    def __del__(self) -> None:
        """Cleanup when the client is destroyed."""
        try:
            self.disconnect()
        except Exception as e:
            # Avoid errors during garbage collection
            logger.debug(f"Error during cleanup: {e}")
            pass

    @operation_timeout()
    def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request to the FAISSx server and handle the response.

        This method handles the low-level communication with the server, including:
        - Adding authentication data
        - Serializing the request with msgpack
        - Sending the request and receiving the response
        - Handling timeouts and network errors
        - Deserializing and validating the response

        The @operation_timeout decorator enforces the configured timeout.

        Args:
            request: Dictionary containing the request parameters

        Returns:
            Dictionary containing the server response

        Raises:
            TimeoutError: If the request times out
            RuntimeError: For other failures (with specific error messages)
        """
        # Verify connection is active
        if not self.socket:
            raise RuntimeError("No active connection. Call connect() first.")

        # Add authentication and tenant information if available
        if self.api_key:
            request["api_key"] = self.api_key
        if self.tenant_id:
            request["tenant_id"] = self.tenant_id

        try:
            # Send request and handle response
            action = request.get("action", "unknown")
            logger.debug(f"Sending request: {action}")

            # Prepare request data - serialize with msgpack
            try:
                encoded_request = msgpack.packb(request)
            except Exception as e:
                logger.error(f"Failed to encode request: {e}")
                raise RuntimeError(f"Failed to encode request: {e}")

            # Send request over ZeroMQ
            try:
                self.socket.send(encoded_request)
            except zmq.ZMQError as e:
                logger.error(f"Failed to send request: {e}")
                self.disconnect()  # Clean up connection on failure
                raise RuntimeError(f"Failed to send request: {e}")

            try:
                # Receive and decode response - will timeout based on socket settings
                response = self.socket.recv()
                result = msgpack.unpackb(response, raw=False)

                # Check for server-side errors in response
                if not result.get("success", False) and "error" in result:
                    error_msg = result.get("error", "Unknown server error")
                    logger.error(f"Server returned error: {error_msg}")
                    raise RuntimeError(f"FAISSx server error: {error_msg}")

                logger.debug(f"Request successful: {action}")
                return result

            except zmq.Again:
                # Handle timeout - zmq.Again is raised when recv times out
                logger.error(f"Request timed out after {self.timeout}s: {action}")
                self.disconnect()  # Clean up connection on timeout
                raise TimeoutError(f"Request timed out after {self.timeout}s")

        except TimeoutError:
            # Propagate timeout error after cleanup
            self.disconnect()
            raise

        except zmq.ZMQError as e:
            # Handle ZMQ-specific errors
            logger.error(f"ZMQ error: {str(e)}")
            self.disconnect()
            raise RuntimeError(f"ZMQ communication error: {str(e)}")

        except msgpack.PackException as e:
            # Handle message encoding/decoding errors
            logger.error(f"Message encoding/decoding error: {str(e)}")
            self.disconnect()
            raise RuntimeError(f"Message format error: {str(e)}")

        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error during request: {str(e)}")
            self.disconnect()
            raise RuntimeError(f"Request failed: {str(e)}")

    def _prepare_vectors(self, vectors: VectorData) -> List[List[float]]:
        """Convert vector data to a serializable format for transmission.

        This utility method transforms vector data from various input formats (numpy arrays
        or lists) into a consistent, serializable format for transmission to the FAISSx server.
        The method handles:

        1. Type conversion: Ensures numpy arrays are converted to Python lists
        2. Data validation: Checks that the input can be processed as vector data
        3. Format standardization: Produces a uniform output format regardless of input type

        The transformation is necessary because:
        - msgpack cannot directly serialize numpy arrays
        - ZeroMQ requires data in a serializable format
        - The server expects a consistent format for vectors

        Args:
            vectors: Input vectors as numpy array or list of floats/vectors
                    Can be single or multiple vectors of any dimension

        Returns:
            A nested list representation of the vectors in format [[v1], [v2], ...]
            where each inner list represents one vector

        Raises:
            ValueError: If the input cannot be converted to the expected format,
                       with detailed error information
        """
        try:
            # First ensure input is a numpy array for consistent handling
            if not isinstance(vectors, np.ndarray):
                # Convert to numpy array if it's not already one
                vectors = np.array(vectors, dtype=np.float32)

            # Convert to list format for serialization
            # numpy arrays can't be directly serialized with msgpack
            result = vectors.tolist() if hasattr(vectors, "tolist") else vectors

            # Validate the result is a list (either of vectors or a single vector)
            if not isinstance(result, list):
                raise ValueError(f"Expected list, got {type(result)}")

            return result
        except Exception as e:
            # Provide detailed error for debugging
            logger.error(f"Failed to prepare vectors: {e}")
            raise ValueError(f"Invalid vector format: {e}")

    def create_index(self, name: str, dimension: int, index_type: str = "L2") -> IndexID:
        """Create a new vector index on the server.

        This method establishes a new index instance on the FAISSx server with the
        specified configuration. After creation, the index is immediately ready for
        adding vectors and performing searches.

        Index types determine the search algorithm and performance characteristics:
        - "L2": Flat L2 index with exact Euclidean distance (slow but 100% accurate)
        - "IP": Flat IP index with inner product similarity (for cosine similarity)
        - "HNSW": Hierarchical navigable small world graph (fast approximate search)
        - "IVF": Inverted file index (clustered search for better performance)
        - "IVFPQ": IVF with product quantization (compact storage, faster search)
        - "SQ": Scalar quantization (memory-efficient storage)

        In remote mode, this operation creates the index on the server. In local mode,
        this would be handled by the specific index class constructors instead.

        Args:
            name: Unique identifier/name for the index
                Should be unique across the server to avoid conflicts
            dimension: Dimensionality of the vectors to be indexed
                Must match the vector size used when adding vectors
            index_type: Type of index to create (default: "L2")
                Determines search algorithm and performance characteristics

        Returns:
            Index identifier (string) used for subsequent operations
            Note: This may be the same as 'name' or a server-generated ID

        Raises:
            RuntimeError: If index creation fails on the server, with error details
            ValueError: If dimension or index_type are invalid

        Example:
            ```python
            client = get_client()
            # Create a 128-dimensional flat L2 index
            index_id = client.create_index("customer-embeddings", 128, "L2")
            ```
        """
        logger.info(f"Creating index '{name}' with dimension {dimension}, type {index_type}")

        # Send index creation request
        response = self._send_request(
            {
                "action": "create_index",
                "index_id": name,
                "dimension": dimension,
                "index_type": index_type,
            }
        )

        # Extract the index ID from the response
        index_id = response.get("index_id", name)
        logger.info(f"Created index with ID: {index_id}")
        return index_id

    def add_vectors(self, index_id: IndexID, vectors: VectorData) -> SearchResult:
        """Add vectors to an existing index.

        This method adds new vectors to the specified index. The vectors will
        be available for searching immediately after successful addition.

        Args:
            index_id: Identifier of the target index
            vectors: Numpy array or list of vectors to add

        Returns:
            Dictionary containing operation results and success status:
            {
                "success": bool,      # Whether the operation succeeded
                "count": int,         # Number of vectors added
                "total": int          # Total number of vectors in the index after addition
            }

        Raises:
            ValueError: If the vector format is invalid
            RuntimeError: If adding vectors fails on the server
        """
        try:
            # Log the operation
            logger.debug(f"Adding {len(vectors)} vectors to index {index_id}")

            # Prepare vectors for transmission
            prepared_vectors = self._prepare_vectors(vectors)

            # Send request to add vectors
            result = self._send_request(
                {
                    "action": "add_vectors",
                    "index_id": index_id,
                    "vectors": prepared_vectors,
                }
            )

            # Log success and return result
            logger.debug(f"Added {result.get('count', 0)} vectors to index {index_id}")
            return result

        except Exception as e:
            # Provide detailed error information
            logger.error(f"Failed to add vectors to index {index_id}: {e}")
            raise

    def batch_add_vectors(
        self, index_id: IndexID, vectors: VectorData, batch_size: int = DEFAULT_BATCH_SIZE_ADD
    ) -> SearchResult:
        """Add vectors to an index in batches to handle large datasets efficiently.

        This method splits large datasets into manageable batches to avoid overwhelming
        the server with a single large request. It tracks progress across batches
        and handles failures gracefully.

        Args:
            index_id: Identifier of the target index
            vectors: Numpy array or list of vectors to add
            batch_size: Number of vectors per batch (default: 1000)

        Returns:
            Dictionary containing operation results, success status, and statistics:
            {
                "success": bool,  # Whether the operation succeeded
                "count": int,     # Number of vectors successfully added
                "total": int      # Total vectors in the index after addition
            }

        Raises:
            ValueError: If the vector format is invalid
            RuntimeError: If adding vectors fails on the server
        """
        # Convert input to numpy array for consistent batch handling
        if not isinstance(vectors, np.ndarray):
            try:
                vectors = np.array(vectors, dtype=np.float32)
            except Exception as e:
                raise ValueError(f"Failed to convert input to numpy array: {e}")

        # Initialize tracking variables for batch processing
        total_vectors = vectors.shape[0]
        total_added = 0
        results = {"success": True, "count": 0, "total": 0}

        logger.info(
            f"Batch adding {total_vectors} vectors to index {index_id} "
            f"with batch size {batch_size}"
        )

        # Process vectors in batches
        for i in range(0, total_vectors, batch_size):
            # Extract current batch using slicing
            batch = vectors[i:min(i + batch_size, total_vectors)]
            batch_size_actual = len(batch)
            batch_num = i//batch_size + 1

            logger.debug(f"Processing batch {batch_num}: {batch_size_actual} vectors")

            try:
                # Add the current batch of vectors
                batch_result = self.add_vectors(index_id, batch)
            except Exception as e:
                # Handle unexpected errors during batch processing
                logger.error(f"Batch add failed at offset {i}: {e}")
                return {
                    "success": False,
                    "error": f"Failed at offset {i}: {str(e)}",
                    "count": total_added,
                    "total": results.get("total", 0),
                }

            # Handle batch failure reported by server
            if not batch_result.get("success", False):
                error_msg = batch_result.get("error", "Unknown error")
                logger.error(f"Server reported error on batch {batch_num}: {error_msg}")
                return {
                    "success": False,
                    "error": f"Failed at batch {batch_num}: {error_msg}",
                    "count": total_added,
                    "total": batch_result.get("total", 0),
                }

            # Update progress tracking
            added_count = batch_result.get("count", 0)
            total_added += added_count
            results["total"] = batch_result.get("total", 0)

            logger.debug(f"Added {added_count} vectors in batch {batch_num}")

        # All batches processed successfully
        results["count"] = total_added
        logger.info(f"Successfully added {total_added} vectors to index {index_id}")
        return results

    def search(
        self,
        index_id: IndexID,
        query_vectors: VectorData,
        k: int = DEFAULT_K,
        params: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """Search for k nearest neighbors of query vectors in the specified index.

        This method performs vector similarity search, finding the k closest vectors
        to each query vector according to the index's distance metric.

        Args:
            index_id: Identifier of the target index
            query_vectors: Query vectors to search for
            k: Number of nearest neighbors to return (default: 10)
            params: Additional search parameters (optional)
                Possible parameters depend on the index type and include:
                - "ef" for HNSW indices: controls search accuracy vs speed
                - "nprobe" for IVF indices: controls number of clusters to search

        Returns:
            Dictionary containing search results and success status:
            {
                "success": bool,
                "results": [
                    {
                        "distances": [float, ...],  # Distance values (k entries)
                        "indices": [int, ...]       # Vector indices (k entries)
                    },
                    ...  # One entry per query vector
                ]
            }

        Raises:
            ValueError: If query vector format is invalid
            RuntimeError: If search operation fails on the server
        """
        try:
            # Log the operation
            logger.debug(f"Searching index {index_id} with k={k}")

            # Prepare query vectors for transmission
            prepared_vectors = self._prepare_vectors(query_vectors)

            # Construct the search request
            request = {
                "action": "search",
                "index_id": index_id,
                "query_vectors": prepared_vectors,
                "k": k,
            }

            # Add optional parameters if provided
            if params:
                request["params"] = params

            # Send search request and get results
            result = self._send_request(request)

            # Validate response structure
            if "results" not in result:
                logger.warning(f"Search response missing 'results' field: {result}")

            return result

        except Exception as e:
            # Provide detailed error information
            logger.error(f"Search failed on index {index_id}: {e}")
            raise

    def batch_search(
        self,
        index_id: IndexID,
        query_vectors: VectorData,
        k: int = DEFAULT_K,
        batch_size: int = DEFAULT_BATCH_SIZE_SEARCH,
        params: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """Perform batched nearest neighbor search for optimal performance and reliability.

        This method implements an optimized search strategy for large query sets by:

        1. Automatically splitting the workload into smaller batches
        2. Processing each batch independently with controlled resource usage
        3. Combining the results into a unified response structure

        Batching provides several critical advantages:
        - Prevents timeouts on large search operations
        - Keeps memory usage under control
        - Allows partial progress tracking
        - Improves error isolation (failure in one batch doesn't fail all)
        - Enables better load distribution on the server

        Performance tuning:
        - Increase batch_size for better throughput (if server can handle it)
        - Decrease batch_size if experiencing timeouts or memory issues
        - Consider the parameter in relation to query vector dimensionality

        Args:
            index_id: Identifier of the target index
            query_vectors: Query vectors to search for (numpy array or list)
            k: Number of nearest neighbors to return per query (default: 10)
            batch_size: Number of query vectors per batch (default: 100)
                - Critical for performance tuning
                - Lower values use less memory but have higher overhead
                - Higher values improve throughput but may cause timeouts
            params: Additional search parameters (optional)
                Common parameters include:
                - "nprobe": Number of cells to visit for IVF indices (higher = more accurate)
                - "ef_search": Search depth for HNSW indices (higher = more accurate)

        Returns:
            Dictionary with combined search results and metadata:
            {
                "success": bool,                       # Overall operation success
                "results": [                           # List of results, one per query vector
                    {
                        "distances": [float, ...],     # Distance values (k entries)
                        "indices": [int, ...]          # Vector indices (k entries)
                    },
                    ...
                ]
            }

        Raises:
            ValueError: If query vector format is invalid
            RuntimeError: If search operation fails on the server, with details
                         about which batch failed and why
        """
        # Convert to numpy array for consistent batch handling
        if not isinstance(query_vectors, np.ndarray):
            try:
                query_vectors = np.array(query_vectors, dtype=np.float32)
            except Exception as e:
                raise ValueError(f"Failed to convert query vectors to numpy array: {e}")

        # Initialize tracking
        total_queries = query_vectors.shape[0]
        all_results = []

        # Log the start of the batch search operation
        logger.info(
            f"Batch searching {total_queries} queries in index {index_id} "
            f"with k={k}, batch_size={batch_size}"
        )

        # Process queries in batches
        for i in range(0, total_queries, batch_size):
            # Extract current batch of vectors using slicing
            batch = query_vectors[i:min(i + batch_size, total_queries)]
            batch_size_actual = len(batch)
            batch_num = i//batch_size + 1

            # Log the current batch being processed
            logger.debug(
                f"Processing search batch {batch_num}: "
                f"{batch_size_actual} queries"
            )

            try:
                # Search the current batch
                batch_result = self.search(index_id, batch, k, params)
            except Exception as e:
                # Handle unexpected errors during batch processing
                logger.error(f"Batch search failed at offset {i}: {e}")
                return {
                    "success": False,
                    "error": f"Failed at offset {i}: {str(e)}",
                }

            # Handle batch failure reported by server
            if not batch_result.get("success", False):
                error_msg = batch_result.get("error", "Unknown error")
                logger.error(
                    f"Batch search failed at batch {batch_num}: {error_msg}"
                )
                return {
                    "success": False,
                    "error": f"Failed at batch {batch_num}: {error_msg}",
                }

            # Accumulate results from successful batch
            batch_results = batch_result.get("results", [])
            all_results.extend(batch_results)

            # Log progress
            logger.debug(
                f"Completed batch {batch_num} with {len(batch_results)} results"
            )

        # All batches processed successfully
        logger.info(
            f"Successfully searched {total_queries} queries, "
            f"found {len(all_results)} results"
        )
        return {"success": True, "results": all_results}

    def range_search(
        self, index_id: IndexID, query_vectors: VectorData, radius: float
    ) -> SearchResult:
        """Search for vectors within specified radius of query vectors.

        Args:
            index_id: Identifier of the target index
            query_vectors: Query vectors to search around
            radius: Maximum distance threshold for results

        Returns:
            Dictionary containing vectors within radius and their distances

        Raises:
            ValueError: If query vector format is invalid
            RuntimeError: If search operation fails on the server
        """
        try:
            # Prepare and send range search request
            logger.debug(
                f"Range searching index {index_id} with radius {radius}"
            )

            prepared_vectors = self._prepare_vectors(query_vectors)
            result = self._send_request(
                {
                    "action": "range_search",
                    "index_id": index_id,
                    "query_vectors": prepared_vectors,
                    "radius": float(radius),
                }
            )

            # Validate response format
            if "results" not in result:
                logger.warning(f"Range search missing 'results' field: {result}")

            return result

        except Exception as e:
            logger.error(f"Range search failed on index {index_id}: {e}")
            raise

    def batch_range_search(
        self,
        index_id: IndexID,
        query_vectors: VectorData,
        radius: float,
        batch_size: int = DEFAULT_BATCH_SIZE_SEARCH,
    ) -> SearchResult:
        """Perform batched range search for large query sets efficiently.

        Args:
            index_id: Identifier of the target index
            query_vectors: Query vectors to search around
            radius: Maximum distance threshold for results
            batch_size: Number of queries per batch (default: 100)

        Returns:
            Dictionary containing combined range search results and success status

        Raises:
            ValueError: If query vector format is invalid
            RuntimeError: If search operation fails on the server
        """
        # Ensure input is numpy array with float32 dtype
        if not isinstance(query_vectors, np.ndarray):
            try:
                query_vectors = np.array(query_vectors, dtype=np.float32)
            except Exception as e:
                raise ValueError(
                    f"Failed to convert query vectors to numpy array: {e}"
                )

        total_queries = query_vectors.shape[0]
        all_results = []

        logger.info(
            f"Batch range searching {total_queries} queries in index {index_id} "
            f"with radius={radius}, batch_size={batch_size}"
        )

        # Process queries in batches
        for i in range(0, total_queries, batch_size):
            # Extract current batch of vectors
            batch = query_vectors[i:min(i + batch_size, total_queries)]
            batch_size_actual = len(batch)

            logger.debug(
                f"Processing range search batch {i//batch_size + 1}: "
                f"{batch_size_actual} queries"
            )

            try:
                batch_result = self.range_search(index_id, batch, radius)
            except Exception as e:
                logger.error(f"Batch range search failed at offset {i}: {e}")
                return {
                    "success": False,
                    "error": f"Failed at offset {i}: {str(e)}",
                }

            # Handle batch failure
            if not batch_result.get("success", False):
                error_msg = batch_result.get("error", "Unknown error")
                logger.error(
                    f"Batch range search failed at batch {i//batch_size + 1}: {error_msg}"
                )
                return {
                    "success": False,
                    "error": f"Failed at batch {i//batch_size + 1}: {error_msg}",
                }

            # Accumulate results from successful batch
            batch_results = batch_result.get("results", [])
            all_results.extend(batch_results)

            logger.debug(
                f"Completed batch {i//batch_size + 1} with {len(batch_results)} results"
            )

        logger.info(
            f"Successfully range searched {total_queries} queries, "
            f"found {len(all_results)} results"
        )
        return {"success": True, "results": all_results}

    def get_index_stats(self, index_id: IndexID) -> SearchResult:
        """Get statistics and metadata for specified index.

        Args:
            index_id: Identifier of the target index

        Returns:
            Dictionary containing index statistics (dimension, vector count, etc.)

        Raises:
            RuntimeError: If the server returns an error
        """
        try:
            # Request index statistics from server
            logger.debug(f"Getting stats for index {index_id}")

            result = self._send_request(
                {
                    "action": "get_index_stats",
                    "index_id": index_id,
                }
            )

            logger.debug(f"Retrieved stats for index {index_id}")
            return result

        except Exception as e:
            logger.error(f"Failed to get stats for index {index_id}: {e}")
            raise

    def list_indexes(self) -> SearchResult:
        """List all available indexes on the server.

        Returns:
            Dictionary containing list of indexes and their metadata

        Raises:
            RuntimeError: If the server returns an error
        """
        try:
            # Request list of all indexes from server
            logger.debug("Listing all indexes")

            result = self._send_request({"action": "list_indexes"})

            index_count = len(result.get("indexes", []))
            logger.debug(f"Retrieved list of {index_count} indexes")
            return result

        except Exception as e:
            logger.error(f"Failed to list indexes: {e}")
            raise

    def train_index(
        self, index_id: IndexID, training_vectors: VectorData
    ) -> SearchResult:
        """Train an index with the provided vectors (required for IVF indices).

        Args:
            index_id: ID of the index to train
            training_vectors: Vectors to use for training

        Returns:
            Dictionary containing training results and success status

        Raises:
            ValueError: If training vector format is invalid
            RuntimeError: If training fails on the server
        """
        try:
            # Prepare vectors
            vector_count = (
                len(training_vectors)
                if hasattr(training_vectors, "__len__")
                else "unknown"
            )
            logger.info(f"Training index {index_id} with {vector_count} vectors")

            prepared_vectors = self._prepare_vectors(training_vectors)

            # Send training request with prepared vectors
            result = self._send_request(
                {
                    "action": "train_index",
                    "index_id": index_id,
                    "training_vectors": prepared_vectors,
                }
            )

            logger.info(f"Successfully trained index {index_id}")
            return result

        except Exception as e:
            logger.error(f"Failed to train index {index_id}: {e}")
            raise

    def close(self) -> None:
        """Clean up resources and close the connection to the server.

        This method ensures proper cleanup of ZeroMQ resources.
        """
        logger.debug("Closing client connection")
        self.disconnect()


# Global singleton instance of FaissXClient
_client: Optional[FaissXClient] = None


def get_client() -> Optional[FaissXClient]:
    """Get or initialize the singleton FaissXClient instance.

    This function implements the singleton pattern to ensure a single shared
    client instance is used throughout the application. This approach:
    1. Minimizes connection overhead by reusing connections
    2. Ensures consistent configuration across the application
    3. Properly manages resources for ZeroMQ sockets

    Behavior:
    - Creates a new client instance if none exists
    - Returns the existing instance if already initialized
    - Attempts to connect if server address is configured but no connection exists
    - Returns None if no server is configured or connection attempt fails

    Returns:
        The active client instance or None if no client could be initialized or connected

    Example:
        ```python
        from faissx.client import get_client, configure

        # First configure the client
        configure(server="tcp://localhost:45678")

        # Then get the configured client
        client = get_client()
        if client:
            results = client.search(...)
        ```

    Note:
        Most applications should not need to call this directly.
        The index classes automatically use this function internally
        to access the client when needed.
    """
    global _client

    # Initialize client if it doesn't exist yet
    if not _client:
        _client = FaissXClient()

    # Get the configured client or None if not configured
    return _client.get_client()


def configure(
    server: Optional[str] = None,
    api_key: Optional[str] = None,
    tenant_id: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> None:
    """Configure the global FAISSx client with connection parameters.

    This is the main entry point for setting up FAISSx in remote mode. When called,
    it configures the global client instance to connect to a remote FAISSx server.
    If you don't call this function, FAISSx operates in local mode using your
    installed FAISS library.

    IMPORTANT: Once configured for remote operation, all subsequent operations MUST
    use the remote server. There is no automatic fallback to local mode if the server
    is unavailable.

    Args:
        server: ZeroMQ server address (e.g., "tcp://localhost:45678")
            If not provided, uses FAISSX_SERVER environment variable
        api_key: API key for authentication with the server
            If not provided, uses FAISSX_API_KEY environment variable
        tenant_id: Tenant identifier for multi-tenant environments
            If not provided, uses FAISSX_TENANT_ID environment variable
        timeout: Connection timeout in seconds (default: 5.0)
            How long to wait for server responses before timing out

    Returns:
        None

    Raises:
        RuntimeError: If the connection to the server fails and cannot be established

    Example:
        ```python
        # Configure using explicit parameters
        from faissx import client as faiss
        faiss.configure(
            server="tcp://faiss-server:45678",
            api_key="your-api-key",
            tenant_id="your-tenant"
        )

        # Or configure using environment variables
        # os.environ["FAISSX_SERVER"] = "tcp://faiss-server:45678"
        # os.environ["FAISSX_API_KEY"] = "your-api-key"
        # os.environ["FAISSX_TENANT_ID"] = "your-tenant"
        # faiss.configure()

        # After configure(), all operations use the remote server
        index = faiss.IndexFlatL2(128)
        ```
    """
    global _client

    # Initialize client if needed
    get_client()

    # Apply new configuration if client exists
    if _client:
        try:
            # Configure the client with provided settings
            _client.configure(server, api_key, tenant_id, timeout)
            logger.info("Successfully configured global FAISSx client")
        except Exception as e:
            # Log and propagate configuration failures
            logger.error(f"Failed to configure global FAISSx client: {e}")
            raise
