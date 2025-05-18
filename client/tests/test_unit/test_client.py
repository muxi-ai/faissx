import pytest
import os
from unittest.mock import patch, MagicMock

import faiss_proxy
from faiss_proxy.client import FaissProxyClient, configure, get_client


class TestFaissProxyClient:
    """Unit tests for the FAISS Proxy client"""

    def test_client_initialization(self):
        """Test client initialization with configuration"""
        client = FaissProxyClient(
            api_url="http://test-server:8000",
            api_key="test-api-key",
            tenant_id="test-tenant"
        )

        assert client.api_url == "http://test-server:8000"
        assert client.api_key == "test-api-key"
        assert client.tenant_id == "test-tenant"

        # Check that session headers are set correctly
        assert "X-API-Key" in client.session.headers
        assert client.session.headers["X-API-Key"] == "test-api-key"
        assert client.session.headers["Content-Type"] == "application/json"

    def test_client_missing_config(self):
        """Test client initialization with missing configuration"""
        # Should raise ValueError when config is missing
        with pytest.raises(ValueError, match="API URL must be provided"):
            FaissProxyClient(api_url=None, api_key="key", tenant_id="tenant")

        with pytest.raises(ValueError, match="API key must be provided"):
            FaissProxyClient(api_url="url", api_key=None, tenant_id="tenant")

        with pytest.raises(ValueError, match="Tenant ID must be provided"):
            FaissProxyClient(api_url="url", api_key="key", tenant_id=None)

    def test_configure_function(self):
        """Test the configure function for setting global configuration"""
        # Save original values
        original_url = faiss_proxy._API_URL
        original_key = faiss_proxy._API_KEY
        original_tenant = faiss_proxy._TENANT_ID

        try:
            # Configure with new values
            configure(
                api_url="http://new-server:8000",
                api_key="new-api-key",
                tenant_id="new-tenant"
            )

            # Check that module-level variables are updated
            assert faiss_proxy._API_URL == "http://new-server:8000"
            assert faiss_proxy._API_KEY == "new-api-key"
            assert faiss_proxy._TENANT_ID == "new-tenant"
        finally:
            # Restore original values
            faiss_proxy._API_URL = original_url
            faiss_proxy._API_KEY = original_key
            faiss_proxy._TENANT_ID = original_tenant

    @patch("faiss_proxy.client._client", None)
    @patch("faiss_proxy.client.FaissProxyClient")
    def test_get_client(self, mock_client_class):
        """Test the get_client function creates and caches client instance"""
        # Setup mock
        mock_instance = MagicMock()
        mock_client_class.return_value = mock_instance

        # First call should create a new client
        client1 = get_client()
        assert client1 == mock_instance
        mock_client_class.assert_called_once()

        # Second call should return the same instance
        mock_client_class.reset_mock()  # Reset the mock to check it's not called again
        client2 = get_client()
        assert client2 == mock_instance
        mock_client_class.assert_not_called()  # Should not create another instance

    @patch("requests.Session")
    def test_make_request(self, mock_session_class):
        """Test the _make_request method with different HTTP methods"""
        # Setup mock
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}

        mock_session.get.return_value = mock_response
        mock_session.post.return_value = mock_response
        mock_session.delete.return_value = mock_response

        mock_session_class.return_value = mock_session

        # Create client with mocked session
        client = FaissProxyClient(
            api_url="http://test-server:8000",
            api_key="test-api-key",
            tenant_id="test-tenant"
        )
        client.session = mock_session

        # Test GET request
        result = client._make_request("GET", "/test", {"param": "value"})
        assert result == {"result": "success"}
        mock_session.get.assert_called_with(
            "http://test-server:8000/test",
            json={"param": "value"}
        )

        # Test POST request
        result = client._make_request("POST", "/test", {"param": "value"})
        assert result == {"result": "success"}
        mock_session.post.assert_called_with(
            "http://test-server:8000/test",
            json={"param": "value"}
        )

        # Test DELETE request
        result = client._make_request("DELETE", "/test", {"param": "value"})
        assert result == {"result": "success"}
        mock_session.delete.assert_called_with(
            "http://test-server:8000/test",
            json={"param": "value"}
        )

    @patch("requests.Session")
    def test_create_index(self, mock_session_class):
        """Test the create_index method"""
        # Setup mock
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "test-index-id"}

        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session

        # Create client with mocked session
        client = FaissProxyClient(
            api_url="http://test-server:8000",
            api_key="test-api-key",
            tenant_id="test-tenant"
        )
        client.session = mock_session

        # Test create_index
        index_id = client.create_index("test-index", 128, "IndexFlatL2")

        assert index_id == "test-index-id"
        mock_session.post.assert_called_with(
            "http://test-server:8000/v1/index",
            json={
                "name": "test-index",
                "dimension": 128,
                "index_type": "IndexFlatL2",
                "tenant_id": "test-tenant"
            }
        )
