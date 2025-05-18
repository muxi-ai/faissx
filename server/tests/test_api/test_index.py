import pytest
from fastapi import status


class TestIndexAPI:
    """Test cases for index management API"""

    def test_create_index(self, api_client, test_tenant):
        """Test creating a new index"""
        # Prepare data for index creation
        index_data = {
            "name": "test-index",
            "dimension": 128,
            "index_type": "IndexFlatL2",
            "tenant_id": test_tenant
        }

        # Make request to create index
        response = api_client.post("/v1/index", json=index_data)

        # Check response
        assert response.status_code == status.HTTP_201_CREATED

        # Verify response data
        data = response.json()
        assert "id" in data
        assert data["name"] == index_data["name"]
        assert data["dimension"] == index_data["dimension"]
        assert data["index_type"] == index_data["index_type"]
        assert data["tenant_id"] == test_tenant

        # Save index_id for other tests
        return data["id"]

    def test_get_index(self, api_client, test_tenant):
        """Test retrieving index info"""
        # First create an index
        index_id = self.test_create_index(api_client, test_tenant)

        # Get the index info
        response = api_client.get(f"/v1/index/{index_id}")

        # Check response
        assert response.status_code == status.HTTP_200_OK

        # Verify response data
        data = response.json()
        assert data["id"] == index_id
        assert data["tenant_id"] == test_tenant

    def test_delete_index(self, api_client, test_tenant):
        """Test deleting an index"""
        # First create an index
        index_id = self.test_create_index(api_client, test_tenant)

        # Delete the index
        response = api_client.delete(f"/v1/index/{index_id}")

        # Check response
        assert response.status_code == status.HTTP_204_NO_CONTENT

        # Verify the index is gone
        get_response = api_client.get(f"/v1/index/{index_id}")
        assert get_response.status_code == status.HTTP_404_NOT_FOUND

    def test_create_index_invalid_tenant(self, api_client):
        """Test creating an index with invalid tenant ID"""
        # Prepare data with wrong tenant ID
        index_data = {
            "name": "test-index",
            "dimension": 128,
            "index_type": "IndexFlatL2",
            "tenant_id": "wrong-tenant-id"
        }

        # Make request
        response = api_client.post("/v1/index", json=index_data)

        # Check response - should be forbidden
        assert response.status_code == status.HTTP_403_FORBIDDEN
