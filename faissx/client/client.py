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

        # Ensure server address is provided
        if not self.server:
            raise ValueError("Server address must be provided")

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

    def search(
        self,
        index_id: str,
        query_vectors: np.ndarray,
        k: int = 10,
    ) -> Dict[str, Any]:
        """
        Search for similar vectors in an index.

        Args:
            index_id: ID of the index to search
            query_vectors: Query vectors to find matches for
            k: Number of nearest neighbors to return

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

        return self._send_request(request)

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


def get_client() -> FaissXClient:
    """
    Get or create the singleton client instance.

    Returns:
        Configured FaissXClient instance

    Raises:
        RuntimeError: If connection fails or authentication is missing
    """
    global _client

    if _client is None:
        _client = FaissXClient()

    return _client


def __del__():
    """
    Cleanup handler called when the module is unloaded.

    Ensures proper cleanup of the client instance and its resources.
    """
    global _client
    if _client:
        _client.close()
