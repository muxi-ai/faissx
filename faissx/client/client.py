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

import os
import zmq
import msgpack
import numpy as np
import logging
from typing import Dict, Any, Optional
import time
from functools import wraps

# Import the timeout decorator from our custom module
from .timeout import timeout as operation_timeout, TimeoutError, TIMEOUT

# Configure logging for the module
logger = logging.getLogger(__name__)


def retry_on_failure(max_retries=2, delay=1):
    """Decorator that retries a function call on failure.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay in seconds between retries

    Returns:
        Decorated function that will retry on failure
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0
            last_error = None

            while retry_count <= max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    retry_count += 1
                    logger.warning(
                        f"Operation failed (attempt {retry_count}/{max_retries+1}): {e}"
                    )
                    if retry_count <= max_retries:
                        time.sleep(delay)

            raise RuntimeError(
                f"Operation failed after {max_retries+1} attempts: {last_error}"
            )

        return wrapper

    return decorator


class FaissXClient:
    """Client for interacting with FAISSx server via ZeroMQ.

    This client provides methods to create and manage vector indexes, add vectors,
    and perform similarity searches using the FAISS library through a ZeroMQ server.
    """

    def __init__(self):
        """Initialize the client with configuration from environment variables."""
        self.server = os.environ.get("FAISSX_SERVER", "")
        self.api_key = os.environ.get("FAISSX_API_KEY", "")
        self.tenant_id = os.environ.get("FAISSX_TENANT_ID", "")
        self.context = None
        self.socket = None
        self.mode = "local"

    def configure(
        self,
        server: Optional[str] = None,
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
        timeout: float = 5.0,
    ) -> None:
        """Configure the client with server details and authentication.

        Args:
            server: ZeroMQ server address
            api_key: API key for authentication
            tenant_id: Tenant identifier for multi-tenant setups
            timeout: Connection timeout in seconds (default: 5.0)
        """
        global TIMEOUT
        TIMEOUT = timeout

        self.server = server or self.server
        self.api_key = api_key or self.api_key
        self.tenant_id = tenant_id or self.tenant_id
        self.connect()
        self.mode = "remote"

    def disconnect(self):
        """Close the ZeroMQ connection and clean up resources."""
        if self.socket:
            self.socket.close()
            self.context.term()
            self.socket = None
            self.context = None

    @retry_on_failure(max_retries=2)
    def connect(self):
        """Establish connection to the FAISSx server with retry logic."""
        self.disconnect()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.server)
        self._send_request({"action": "ping"})
        logger.info(f"Connected to FAISSx server at {self.server}")

    def get_client(self):
        """Get an active client instance, creating one if necessary."""
        if not self.socket and not self.server:
            return None
        if not self.socket:
            return self.connect()
        return self

    def __del__(self):
        """Cleanup when the client is destroyed."""
        self.disconnect()

    @operation_timeout()
    def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request to the FAISSx server and handle the response.

        Args:
            request: Dictionary containing the request parameters

        Returns:
            Dictionary containing the server response

        Raises:
            RuntimeError: If the request fails or server returns an error
            TimeoutError: If the request times out
        """
        if self.api_key:
            request["api_key"] = self.api_key
        if self.tenant_id:
            request["tenant_id"] = self.tenant_id

        try:
            self.socket.send(msgpack.packb(request))
            response = self.socket.recv()
            result = msgpack.unpackb(response, raw=False)

            if not result.get("success", False) and "error" in result:
                raise RuntimeError(f"FAISSx request failed: {result['error']}")

            return result
        except TimeoutError:
            self.disconnect()  # Clean up resources
            raise  # Re-raise the TimeoutError
        except zmq.ZMQError as e:
            self.disconnect()  # Clean up resources
            raise RuntimeError(f"ZMQ error: {str(e)}")
        except Exception as e:
            self.disconnect()  # Clean up resources
            raise RuntimeError(f"FAISSx request failed: {str(e)}")

    def _prepare_vectors(self, vectors: np.ndarray) -> list:
        """Convert numpy arrays to a format suitable for serialization.

        Args:
            vectors: Input vectors as numpy array

        Returns:
            List representation of the vectors
        """
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)
        return vectors.tolist() if hasattr(vectors, "tolist") else vectors

    def create_index(self, name: str, dimension: int, index_type: str = "L2") -> str:
        """Create a new vector index.

        Args:
            name: Unique identifier for the index
            dimension: Dimensionality of the vectors
            index_type: Type of index (default: "L2")

        Returns:
            Index identifier
        """
        response = self._send_request(
            {
                "action": "create_index",
                "index_id": name,
                "dimension": dimension,
                "index_type": index_type,
            }
        )
        return response.get("index_id", name)

    def add_vectors(self, index_id: str, vectors: np.ndarray) -> Dict[str, Any]:
        """Add vectors to an existing index.

        Args:
            index_id: Identifier of the target index
            vectors: Numpy array of vectors to add

        Returns:
            Dictionary containing operation results
        """
        return self._send_request(
            {
                "action": "add_vectors",
                "index_id": index_id,
                "vectors": self._prepare_vectors(vectors),
            }
        )

    def batch_add_vectors(
        self, index_id: str, vectors: np.ndarray, batch_size: int = 1000
    ) -> Dict[str, Any]:
        """Add vectors to an index in batches.

        Args:
            index_id: Identifier of the target index
            vectors: Numpy array of vectors to add
            batch_size: Number of vectors per batch

        Returns:
            Dictionary containing operation results and statistics
        """
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)

        total_vectors = vectors.shape[0]
        total_added = 0
        results = {"success": True, "count": 0, "total": 0}

        for i in range(0, total_vectors, batch_size):
            batch = vectors[i:min(i + batch_size, total_vectors)]
            batch_result = self.add_vectors(index_id, batch)

            if not batch_result.get("success", False):
                return {
                    "success": False,
                    "error": (
                        f"Failed at batch {i//batch_size}: "
                        f"{batch_result.get('error', 'Unknown error')}"
                    ),
                    "count": total_added,
                    "total": batch_result.get("total", 0),
                }

            total_added += batch_result.get("count", 0)
            results["total"] = batch_result.get("total", 0)

        results["count"] = total_added
        return results

    def search(
        self,
        index_id: str,
        query_vectors: np.ndarray,
        k: int = 10,
        params: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Search for nearest neighbors in the index.

        Args:
            index_id: Identifier of the target index
            query_vectors: Query vectors to search for
            k: Number of nearest neighbors to return
            params: Additional search parameters

        Returns:
            Dictionary containing search results
        """
        request = {
            "action": "search",
            "index_id": index_id,
            "query_vectors": self._prepare_vectors(query_vectors),
            "k": k,
        }
        if params:
            request["params"] = params
        return self._send_request(request)

    def batch_search(
        self,
        index_id: str,
        query_vectors: np.ndarray,
        k: int = 10,
        batch_size: int = 100,
        params: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Perform batched nearest neighbor search for large query sets.

        Args:
            index_id: Identifier of the target index
            query_vectors: Query vectors to search for
            k: Number of nearest neighbors to return
            batch_size: Number of queries per batch
            params: Additional search parameters

        Returns:
            Dictionary containing combined search results and success status
        """
        # Convert input to numpy array if needed
        if not isinstance(query_vectors, np.ndarray):
            query_vectors = np.array(query_vectors, dtype=np.float32)

        total_queries = query_vectors.shape[0]
        all_results = []

        # Process queries in batches
        for i in range(0, total_queries, batch_size):
            # Extract current batch of vectors
            batch = query_vectors[i:min(i + batch_size, total_queries)]
            batch_result = self.search(index_id, batch, k, params)

            # Handle batch failure
            if not batch_result.get("success", False):
                return {
                    "success": False,
                    "error": (
                        f"Failed at batch {i//batch_size}: "
                        f"{batch_result.get('error', 'Unknown error')}"
                    ),
                }

            # Accumulate results
            all_results.extend(batch_result.get("results", []))

        return {"success": True, "results": all_results}

    def range_search(
        self, index_id: str, query_vectors: np.ndarray, radius: float
    ) -> Dict[str, Any]:
        """Search for vectors within specified radius of query vectors.

        Args:
            index_id: Identifier of the target index
            query_vectors: Query vectors to search around
            radius: Maximum distance threshold for results

        Returns:
            Dictionary containing vectors within radius and their distances
        """
        return self._send_request(
            {
                "action": "range_search",
                "index_id": index_id,
                "query_vectors": self._prepare_vectors(query_vectors),
                "radius": float(radius),
            }
        )

    def batch_range_search(
        self,
        index_id: str,
        query_vectors: np.ndarray,
        radius: float,
        batch_size: int = 100,
    ) -> Dict[str, Any]:
        """Perform batched range search for large query sets.

        Args:
            index_id: Identifier of the target index
            query_vectors: Query vectors to search around
            radius: Maximum distance threshold for results
            batch_size: Number of queries per batch

        Returns:
            Dictionary containing combined range search results
        """
        # Convert input to numpy array if needed
        if not isinstance(query_vectors, np.ndarray):
            query_vectors = np.array(query_vectors, dtype=np.float32)

        total_queries = query_vectors.shape[0]
        all_results = []

        # Process queries in batches
        for i in range(0, total_queries, batch_size):
            # Extract current batch of vectors
            batch = query_vectors[i:min(i + batch_size, total_queries)]
            batch_result = self.range_search(index_id, batch, radius)

            # Handle batch failure
            if not batch_result.get("success", False):
                return {
                    "success": False,
                    "error": (
                        f"Failed at batch {i//batch_size}: "
                        f"{batch_result.get('error', 'Unknown error')}"
                    ),
                }

            # Accumulate results
            all_results.extend(batch_result.get("results", []))

        return {"success": True, "results": all_results}

    def get_index_stats(self, index_id: str) -> Dict[str, Any]:
        """Get statistics and metadata for specified index.

        Args:
            index_id: Identifier of the target index

        Returns:
            Dictionary containing index statistics (dimension, vector count, etc.)
        """
        return self._send_request(
            {
                "action": "get_index_stats",
                "index_id": index_id,
            }
        )

    def list_indexes(self) -> Dict[str, Any]:
        """List all available indexes on the server.

        Returns:
            Dictionary containing list of indexes and their metadata
        """
        return self._send_request({"action": "list_indexes"})

    def train_index(
        self, index_id: str, training_vectors: np.ndarray
    ) -> Dict[str, Any]:
        """Train an index with the provided vectors (required for IVF indices).

        Args:
            index_id: ID of the index to train
            training_vectors: Vectors to use for training

        Returns:
            Dictionary containing training results
        """
        return self._send_request(
            {
                "action": "train_index",
                "index_id": index_id,
                "training_vectors": self._prepare_vectors(training_vectors),
            }
        )

    def close(self) -> None:
        """Clean up resources and close the connection."""
        self.disconnect()


# Global singleton instance of FaissXClient
_client: Optional[FaissXClient] = None


def get_client() -> FaissXClient:
    """Initialize or retrieve the singleton FaissXClient instance.

    Creates a new client instance if none exists, otherwise returns the existing one.
    This ensures we maintain a single client connection throughout the application.

    Returns:
        FaissXClient: The active client instance
    """
    global _client
    if not _client:
        _client = FaissXClient()
    return _client.get_client()


def configure(
    server: Optional[str] = None,
    api_key: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> None:
    """Configure the global FaissXClient instance with server and auth settings.

    Updates the client configuration with new server address, API key, and tenant ID.
    Creates a new client instance if one doesn't exist.

    Args:
        server: ZeroMQ server address (e.g. "tcp://localhost:45678")
        api_key: API key for server authentication
        tenant_id: Tenant identifier for multi-tenant isolation
    """
    global _client

    # Initialize client if needed and apply new configuration
    get_client()
    _client.configure(server, api_key, tenant_id)
