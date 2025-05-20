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
FAISSx ZeroMQ Client Implementation

This module provides the core client interface for communicating with
the FAISSx vector server.

It handles:
- ZeroMQ socket communication with binary protocol support
- Request/response cycles for all vector operations
- Authentication and tenant isolation
- Connection management and error handling
- Serialization/deserialization of vector data
- Index creation, vector addition, and similarity searches

The FaissXClient class handles low-level communication details, while the
public configure() and get_client() functions provide a simplified interface
for the rest of the client library.
"""

import zmq
import msgpack
import numpy as np
import logging
from typing import Dict, Any, Optional
import time

# Configure logging for the module
logger = logging.getLogger(__name__)


class FaissXClient:
    """
    Client for interacting with FAISSx server via ZeroMQ.

    This class handles all communication with the FAISSx server, including:
    - Connection management
    - Request/response handling
    - Vector operations (create, add, search)
    - Index management
    """

    def __init__(
        self,
        server: Optional[str] = None,
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ):
        """
        Initialize the client with server connection details and authentication.

        Args:
            server: Server address in ZeroMQ format (e.g. "tcp://localhost:45678")
            api_key: API key for authentication with the server
            tenant_id: Tenant ID for multi-tenant data isolation

        Raises:
            ValueError: If server address is not provided
            RuntimeError: If connection to server fails
        """
        from . import _API_URL, _API_KEY, _TENANT_ID

        # Use provided values or fall back to module defaults
        self.server = server or _API_URL
        self.api_key = api_key or _API_KEY
        self.tenant_id = tenant_id or _TENANT_ID

        # If server address is empty, this is an error - it should be caught by get_client
        # to enable local mode
        if not self.server:
            raise ValueError("Server address is empty, using local mode instead")

        # Set up ZeroMQ connection
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)  # Request-Reply pattern
        self.socket.connect(self.server)

        # Verify connection with a ping request
        try:
            self._send_request({"action": "ping"})
            logger.info(f"Connected to FAISSx server at {self.server}")
        except Exception as e:
            logger.error(f"Failed to connect to FAISSx server: {e}")
            raise RuntimeError(f"Failed to connect to FAISSx server at {self.server}: {e}")

    def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal method to send requests to the server and handle responses.

        Args:
            request: Dictionary containing the request data

        Returns:
            Dictionary containing the server's response

        Raises:
            RuntimeError: If request fails or server returns an error
        """
        # Add authentication headers if configured
        if self.api_key:
            request["api_key"] = self.api_key
        if self.tenant_id:
            request["tenant_id"] = self.tenant_id

        try:
            # Serialize request using msgpack for efficient binary transfer
            self.socket.send(msgpack.packb(request))

            # Wait for and deserialize response
            response = self.socket.recv()
            result = msgpack.unpackb(response, raw=False)

            # Handle error responses
            if not result.get("success", False) and "error" in result:
                logger.error(f"FAISSx request failed: {result['error']}")
                raise RuntimeError(f"FAISSx request failed: {result['error']}")

            return result
        except zmq.ZMQError as e:
            logger.error(f"ZMQ error: {str(e)}")
            raise RuntimeError(f"ZMQ error: {str(e)}")
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            raise RuntimeError(f"FAISSx request failed: {str(e)}")

    def create_index(
        self, name: str, dimension: int, index_type: str = "L2"
    ) -> str:
        """
        Create a new vector index on the server.

        Args:
            name: Unique identifier for the index
            dimension: Dimensionality of vectors to be stored
            index_type: Type of similarity metric
                        ("L2" for Euclidean distance or "IP" for inner product)

        Returns:
            The created index ID (same as name if successful)
        """
        request = {
            "action": "create_index",
            "index_id": name,
            "dimension": dimension,
            "index_type": index_type
        }

        response = self._send_request(request)
        return response.get("index_id", name)

    def add_vectors(self, index_id: str, vectors: np.ndarray) -> Dict[str, Any]:
        """
        Add vectors to an existing index.

        Args:
            index_id: ID of the target index
            vectors: Numpy array of vectors to add

        Returns:
            Dictionary containing operation results and statistics
        """
        # Convert numpy array to list for serialization
        vectors_list = vectors.tolist() if hasattr(vectors, 'tolist') else vectors

        request = {
            "action": "add_vectors",
            "index_id": index_id,
            "vectors": vectors_list
        }

        return self._send_request(request)

    def batch_add_vectors(
        self,
        index_id: str,
        vectors: np.ndarray,
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Add vectors to an index in optimized batches.

        This method improves performance when adding large numbers of vectors
        by splitting them into batches of optimal size for network transmission.

        Args:
            index_id: ID of the target index
            vectors: Numpy array of vectors to add
            batch_size: Size of each batch (default: 1000 vectors)

        Returns:
            Dictionary containing aggregated operation results and statistics
        """
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)

        total_vectors = vectors.shape[0]
        total_added = 0
        results = {"success": True, "count": 0, "total": 0}

        # Process vectors in batches
        for i in range(0, total_vectors, batch_size):
            batch = vectors[i:min(i+batch_size, total_vectors)]

            # Add this batch
            batch_result = self.add_vectors(index_id, batch)

            # If any batch fails, mark the whole operation as failed
            if not batch_result.get("success", False):
                error_msg = batch_result.get('error', 'Unknown error')
                return {
                    "success": False,
                    "error": f"Failed at batch {i//batch_size}: {error_msg}",
                    "count": total_added,
                    "total": batch_result.get("total", 0)
                }

            # Update counters
            total_added += batch_result.get("count", 0)
            results["total"] = batch_result.get("total", 0)  # Get latest total

        # Return aggregated results
        results["count"] = total_added
        return results

    def search(
        self,
        index_id: str,
        query_vectors: np.ndarray,
        k: int = 10,
        params: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Search for similar vectors in an index.

        Args:
            index_id: ID of the index to search
            query_vectors: Query vectors to find matches for
            k: Number of nearest neighbors to return
            params: Additional search parameters (e.g., nprobe for IVF indices)

        Returns:
            Dictionary containing search results and distances
        """
        # Convert numpy array to list for serialization
        vectors_list = (
            query_vectors.tolist() if hasattr(query_vectors, 'tolist') else query_vectors
        )

        request = {
            "action": "search",
            "index_id": index_id,
            "query_vectors": vectors_list,
            "k": k
        }

        # Add search parameters if provided
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
        """
        Search for similar vectors in batches for improved performance.

        For large numbers of query vectors, this method splits the search
        into optimally sized batches to improve network efficiency.

        Args:
            index_id: ID of the index to search
            query_vectors: Query vectors to find matches for
            k: Number of nearest neighbors to return
            batch_size: Size of each batch (default: 100 queries)
            params: Additional search parameters (e.g., nprobe for IVF indices)

        Returns:
            Dictionary containing combined search results and distances
        """
        if not isinstance(query_vectors, np.ndarray):
            query_vectors = np.array(query_vectors, dtype=np.float32)

        total_queries = query_vectors.shape[0]
        all_results = []

        # Process queries in batches
        for i in range(0, total_queries, batch_size):
            batch = query_vectors[i:min(i+batch_size, total_queries)]

            # Search this batch
            batch_result = self.search(index_id, batch, k, params)

            # Check for errors
            if not batch_result.get("success", False):
                error_msg = batch_result.get('error', 'Unknown error')
                return {
                    "success": False,
                    "error": f"Failed at batch {i//batch_size}: {error_msg}"
                }

            # Add results from this batch to our collection
            all_results.extend(batch_result.get("results", []))

        # Return combined results
        return {"success": True, "results": all_results}

    def range_search(
        self,
        index_id: str,
        query_vectors: np.ndarray,
        radius: float,
    ) -> Dict[str, Any]:
        """
        Search for vectors within a specified radius.

        Args:
            index_id: ID of the index to search
            query_vectors: Query vectors to find matches for
            radius: Maximum distance threshold for matches

        Returns:
            Dictionary containing search results including distances and indices
            for each point within the radius
        """
        # Convert numpy array to list for serialization
        vectors_list = (
            query_vectors.tolist() if hasattr(query_vectors, 'tolist') else query_vectors
        )

        request = {
            "action": "range_search",
            "index_id": index_id,
            "query_vectors": vectors_list,
            "radius": float(radius)  # Ensure radius is a float
        }

        return self._send_request(request)

    def batch_range_search(
        self,
        index_id: str,
        query_vectors: np.ndarray,
        radius: float,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Perform radius search in batches for improved performance.

        For large numbers of query vectors, this method splits the range search
        into optimally sized batches to improve network efficiency.

        Args:
            index_id: ID of the index to search
            query_vectors: Query vectors to find matches for
            radius: Maximum distance threshold for matches
            batch_size: Size of each batch (default: 100 queries)

        Returns:
            Dictionary containing combined search results
        """
        if not isinstance(query_vectors, np.ndarray):
            query_vectors = np.array(query_vectors, dtype=np.float32)

        total_queries = query_vectors.shape[0]
        all_results = []

        # Process queries in batches
        for i in range(0, total_queries, batch_size):
            batch = query_vectors[i:min(i+batch_size, total_queries)]

            # Range search this batch
            batch_result = self.range_search(index_id, batch, radius)

            # Check for errors
            if not batch_result.get("success", False):
                error_msg = batch_result.get('error', 'Unknown error')
                return {
                    "success": False,
                    "error": f"Failed at batch {i//batch_size}: {error_msg}"
                }

            # Add results from this batch to our collection
            all_results.extend(batch_result.get("results", []))

        # Return combined results
        return {"success": True, "results": all_results}

    def get_index_stats(self, index_id: str) -> Dict[str, Any]:
        """
        Retrieve statistics about an index.

        Args:
            index_id: ID of the index to get stats for

        Returns:
            Dictionary containing index statistics (dimension, vector count, etc.)
        """
        request = {
            "action": "get_index_stats",
            "index_id": index_id
        }

        return self._send_request(request)

    def list_indexes(self) -> Dict[str, Any]:
        """
        List all available indexes on the server.

        Returns:
            Dictionary containing list of indexes and their metadata
        """
        request = {
            "action": "list_indexes"
        }

        return self._send_request(request)

    def train_index(self, index_id: str, training_vectors: np.ndarray) -> Dict[str, Any]:
        """
        Train an index with the provided vectors (required for IVF indices).

        Args:
            index_id: ID of the index to train
            training_vectors: Vectors to use for training

        Returns:
            Dictionary containing training results
        """
        # Convert numpy array to list for serialization
        vectors_list = (
            training_vectors.tolist() if hasattr(training_vectors, 'tolist')
            else training_vectors
        )

        request = {
            "action": "train_index",
            "index_id": index_id,
            "training_vectors": vectors_list
        }

        return self._send_request(request)

    def close(self) -> None:
        """
        Clean up ZeroMQ resources and close the connection.

        This method should be called when the client is no longer needed
        to properly free system resources.
        """
        if hasattr(self, 'socket') and self.socket:
            self.socket.close()
        if hasattr(self, 'context') and self.context:
            self.context.term()


# Global singleton client instance
_client: Optional[FaissXClient] = None


def configure(
    server: Optional[str] = None,
    api_key: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> None:
    """
    Configure the global FAISSx client settings.

    This function updates the module-level configuration and resets the client
    instance to ensure it uses the new settings.

    Args:
        server: New server address
        api_key: New API key
        tenant_id: New tenant ID
    """
    global _client

    # Update module-level configuration variables
    if server:
        import faissx
        faissx._API_URL = server

    if api_key:
        import faissx
        faissx._API_KEY = api_key

    if tenant_id:
        import faissx
        faissx._TENANT_ID = tenant_id

    # Reset client to force recreation with new settings
    if _client:
        _client.close()
    _client = None


def get_client() -> Optional[FaissXClient]:
    """
    Get or create the singleton client instance.

    Returns:
        Configured FaissXClient instance or None if local mode is required
            - Returns None when server URL is empty or not configured

    Raises:
        RuntimeError: When connection to the server fails after 3 attempts

    This function will operate in one of two modes:
    1. Local mode: If no server URL is configured (_API_URL is empty)
    2. Remote mode: If a server URL is configured, will retry connection up to 3 times
       and raise an error if all attempts fail - it will never fall back to local mode
    """
    global _client

    # Import here to avoid circular imports
    from . import _API_URL

    # Check if server URL is empty or None - use local mode
    if not _API_URL or _API_URL == "":
        return None

    # Only create client if not already existing
    if _client is None:
        max_retries = 2  # Will try a total of 3 times (initial attempt + 2 retries)
        retry_count = 0
        last_error = None

        while retry_count <= max_retries:
            try:
                # Create a ZeroMQ connection and test it directly
                context = zmq.Context()
                socket = context.socket(zmq.REQ)
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second receive timeout
                socket.setsockopt(zmq.SNDTIMEO, 5000)  # 5 second send timeout
                socket.connect(_API_URL)

                # Send a ping request
                request = {"action": "ping"}
                socket.send(msgpack.packb(request))

                # Wait for response
                response = socket.recv()
                result = msgpack.unpackb(response, raw=False)

                # Clean up socket
                socket.close()
                context.term()

                # If ping was successful, create the client
                if result.get("success", False):
                    # Now create the actual client
                    _client = FaissXClient()
                    logger.info(f"Successfully connected to FAISSx server at {_API_URL}")
                    return _client
                else:
                    raise RuntimeError(
                        f"Server returned error: {result.get('error', 'Unknown error')}")
            except Exception as e:
                last_error = e
                retry_count += 1
                logger.warning(
                    f"Failed to connect to FAISSx server "
                    f"(attempt {retry_count}/{max_retries+1}): {e}"
                )

                # If we haven't reached max retries, wait before trying again
                if retry_count <= max_retries:
                    time.sleep(1)  # Wait 1 second between retry attempts

        # If we get here, all retries have failed - make sure to raise an exception
        logger.error(f"Failed to connect to FAISSx server after {max_retries+1} attempts")
        raise RuntimeError(
            f"Failed to connect to FAISSx server after {max_retries+1} attempts: {last_error}"
        )

    # Return existing client
    return _client


def __del__():
    """
    Cleanup handler called when the module is unloaded.

    Ensures proper cleanup of the client instance and its resources.
    """
    global _client
    if _client:
        _client.close()
