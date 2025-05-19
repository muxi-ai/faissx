"""
FAISSx client implementation
"""

import requests
import logging
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)


class FaissXClient:
    """
    Client for interacting with FAISSx server
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ):
        """
        Initialize the client.

        Args:
            api_url: URL of the FAISSx server
            api_key: API key for authentication
            tenant_id: Tenant ID for multi-tenant isolation
        """
        from . import _API_URL, _API_KEY, _TENANT_ID

        self.api_url = api_url or _API_URL
        self.api_key = api_key or _API_KEY
        self.tenant_id = tenant_id or _TENANT_ID

        # Validate configuration
        if not self.api_url:
            raise ValueError("API URL must be provided")
        if not self.api_key:
            raise ValueError("API key must be provided")
        if not self.tenant_id:
            raise ValueError("Tenant ID must be provided")

        # Initialize session for connection pooling
        self.session = requests.Session()
        self.session.headers.update(
            {"X-API-Key": self.api_key, "Content-Type": "application/json"}
        )

    def _make_request(
        self, method: str, endpoint: str, json_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make a request to the FAISSx server.

        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint
            json_data: JSON data to send

        Returns:
            Response data as dictionary

        Raises:
            RuntimeError: If request fails
        """
        url = f"{self.api_url.rstrip('/')}/{endpoint.lstrip('/')}"

        try:
            if method == "GET":
                response = self.session.get(url, json=json_data)
            elif method == "POST":
                response = self.session.post(url, json=json_data)
            elif method == "DELETE":
                response = self.session.delete(url, json=json_data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()

            if response.status_code == 204:  # No content
                return {}

            return response.json()

        except requests.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            raise RuntimeError(f"FAISSx request failed: {str(e)}")

    def create_index(
        self, name: str, dimension: int, index_type: str = "IndexFlatL2"
    ) -> str:
        """
        Create a new index.

        Args:
            name: Name of the index
            dimension: Vector dimension
            index_type: Type of FAISS index

        Returns:
            Index ID
        """
        data = {
            "name": name,
            "dimension": dimension,
            "index_type": index_type,
            "tenant_id": self.tenant_id,
        }

        response = self._make_request("POST", "/v1/index", data)
        return response["id"]

    def add_vectors(self, index_id: str, vectors: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add vectors to an index.

        Args:
            index_id: Index ID
            vectors: Vector batch data

        Returns:
            Response with success information
        """
        return self._make_request("POST", f"/v1/index/{index_id}/vectors", vectors)

    def search(
        self,
        index_id: str,
        vector: list,
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Search for similar vectors.

        Args:
            index_id: Index ID
            vector: Query vector
            k: Number of results to return
            filter_metadata: Metadata filter

        Returns:
            Search results
        """
        data = {"vector": vector, "k": k, "filter": filter_metadata}

        return self._make_request("POST", f"/v1/index/{index_id}/search", data)

    def delete_vector(self, index_id: str, vector_id: str) -> None:
        """
        Delete a vector from an index.

        Args:
            index_id: Index ID
            vector_id: Vector ID
        """
        self._make_request("DELETE", f"/v1/index/{index_id}/vectors/{vector_id}")

    def delete_index(self, index_id: str) -> None:
        """
        Delete an index.

        Args:
            index_id: Index ID
        """
        self._make_request("DELETE", f"/v1/index/{index_id}")


# Singleton client instance
_client: Optional[FaissXClient] = None


def configure(
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> None:
    """
    Configure the FAISSx client.

    Args:
        api_url: URL of the FAISSx server
        api_key: API key for authentication
        tenant_id: Tenant ID for multi-tenant isolation
    """
    global _client

    # Update module-level configuration
    from . import _API_URL, _API_KEY, _TENANT_ID

    if api_url:
        import faissx

        faissx._API_URL = api_url

    if api_key:
        import faissx

        faissx._API_KEY = api_key

    if tenant_id:
        import faissx

        faissx._TENANT_ID = tenant_id

    # Reset client so it's recreated with new configuration
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
