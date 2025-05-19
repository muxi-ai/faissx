"""
FAISSx client implementation using ZeroMQ
"""

import zmq
import msgpack
import numpy as np
import logging
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)


class FaissXClient:
    """
    Client for interacting with FAISSx server via ZeroMQ
    """

    def __init__(
        self,
        server: Optional[str] = None,
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ):
        """
        Initialize the client.

        Args:
            server: Server address (e.g. "tcp://localhost:45678")
            api_key: API key for authentication
            tenant_id: Tenant ID for multi-tenant isolation
        """
        from . import _API_URL, _API_KEY, _TENANT_ID

        self.server = server or _API_URL
        self.api_key = api_key or _API_KEY
        self.tenant_id = tenant_id or _TENANT_ID

        # Validate configuration
        if not self.server:
            raise ValueError("Server address must be provided")

        # Initialize ZeroMQ socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.server)

        # Test connection with ping
        try:
            self._send_request({"action": "ping"})
            logger.info(f"Connected to FAISSx server at {self.server}")
        except Exception as e:
            logger.error(f"Failed to connect to FAISSx server: {e}")
            raise RuntimeError(f"Failed to connect to FAISSx server at {self.server}: {e}")

    def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a request to the FAISSx server.

        Args:
            request: Request dictionary

        Returns:
            Response data as dictionary

        Raises:
            RuntimeError: If request fails
        """
        # Add authentication if provided
        if self.api_key:
            request["api_key"] = self.api_key
        if self.tenant_id:
            request["tenant_id"] = self.tenant_id

        try:
            # Serialize and send request
            self.socket.send(msgpack.packb(request))

            # Receive and deserialize response
            response = self.socket.recv()
            result = msgpack.unpackb(response, raw=False)

            # Check for error
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
        Create a new index.

        Args:
            name: Name of the index (used as index_id)
            dimension: Vector dimension
            index_type: Type of FAISS index (L2 or IP)

        Returns:
            Index ID
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
        Add vectors to an index.

        Args:
            index_id: Index ID
            vectors: Vector data as numpy array

        Returns:
            Response with success information
        """
        # Ensure vectors are in the correct format
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
        Search for similar vectors.

        Args:
            index_id: Index ID
            query_vectors: Query vectors as numpy array
            k: Number of results to return

        Returns:
            Search results
        """
        # Ensure vectors are in the correct format
        vectors_list = query_vectors.tolist() if hasattr(query_vectors, 'tolist') else query_vectors

        request = {
            "action": "search",
            "index_id": index_id,
            "query_vectors": vectors_list,
            "k": k
        }

        return self._send_request(request)

    def get_index_stats(self, index_id: str) -> Dict[str, Any]:
        """
        Get statistics for an index.

        Args:
            index_id: Index ID

        Returns:
            Index statistics
        """
        request = {
            "action": "get_index_stats",
            "index_id": index_id
        }

        return self._send_request(request)

    def list_indexes(self) -> Dict[str, Any]:
        """
        List all available indexes.

        Returns:
            List of indexes
        """
        request = {
            "action": "list_indexes"
        }

        return self._send_request(request)

    def close(self) -> None:
        """
        Close the connection to the server.
        """
        if hasattr(self, 'socket') and self.socket:
            self.socket.close()
        if hasattr(self, 'context') and self.context:
            self.context.term()


# Singleton client instance
_client: Optional[FaissXClient] = None


def configure(
    server: Optional[str] = None,
    api_key: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> None:
    """
    Configure the FAISSx client.

    Args:
        server: Server address (e.g. "tcp://localhost:45678")
        api_key: API key for authentication
        tenant_id: Tenant ID for multi-tenant isolation
    """
    global _client

    # Update module-level configuration
    if server:
        import faissx
        faissx._API_URL = server

    if api_key:
        import faissx
        faissx._API_KEY = api_key

    if tenant_id:
        import faissx
        faissx._TENANT_ID = tenant_id

    # Reset client so it's recreated with new configuration
    if _client:
        _client.close()
    _client = None


def get_client() -> FaissXClient:
    """
    Get the configured client instance.

    Returns:
        FaissXClient instance
    """
    global _client

    if _client is None:
        _client = FaissXClient()

    return _client


def __del__():
    """
    Clean up resources when the module is unloaded.
    """
    global _client
    if _client:
        _client.close()
