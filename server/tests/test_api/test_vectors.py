import numpy as np
from fastapi import status


class TestVectorAPI:
    """Test cases for vector operations API"""

    def _create_test_index(self, api_client, test_tenant):
        """Helper to create a test index"""
        index_data = {
            "name": "test-vectors-index",
            "dimension": 4,  # Small dimension for testing
            "index_type": "IndexFlatL2",
            "tenant_id": test_tenant
        }
        response = api_client.post("/v1/index", json=index_data)
        assert response.status_code == status.HTTP_201_CREATED
        index_id = response.json()["id"]
        print(f"Created test index with ID: {index_id}")
        return index_id

    def test_add_vectors(self, api_client, test_tenant):
        """Test adding vectors to an index"""
        # Create a test index
        index_id = self._create_test_index(api_client, test_tenant)

        # Prepare vector data
        vectors_data = {
            "vectors": [
                {
                    "id": "vec1",
                    "values": [1.0, 0.0, 0.0, 0.0],
                    "metadata": {"type": "test", "label": "first"}
                },
                {
                    "id": "vec2",
                    "values": [0.0, 1.0, 0.0, 0.0],
                    "metadata": {"type": "test", "label": "second"}
                }
            ]
        }

        # Add vectors
        response = api_client.post(f"/v1/index/{index_id}/vectors", json=vectors_data)
        print(f"Add vectors response status: {response.status_code}")
        if response.status_code != status.HTTP_201_CREATED:
            print(f"Error response: {response.json()}")

        # Check response
        assert response.status_code == status.HTTP_201_CREATED

        # Verify response data
        data = response.json()
        assert data["success"] is True
        assert data["added_count"] == 2
        assert data["failed_count"] == 0

        return index_id

    def test_search_vectors(self, api_client, test_tenant):
        """Test searching for similar vectors"""
        # First add vectors to an index
        index_id = self.test_add_vectors(api_client, test_tenant)

        # Prepare search request
        search_data = {
            "vector": [1.0, 0.1, 0.0, 0.0],  # Close to vec1
            "k": 2
        }

        # Search - use POST for search instead of GET since we need to send a JSON body
        response = api_client.post(f"/v1/index/{index_id}/search", json=search_data)
        print(f"Search response status: {response.status_code}")
        if response.status_code != status.HTTP_200_OK:
            print(f"Error response: {response.json()}")

        # Check response
        assert response.status_code == status.HTTP_200_OK

        # Verify response data
        data = response.json()
        assert "results" in data
        assert len(data["results"]) <= 2  # Should return at most k=2 results

        if len(data["results"]) > 0:
            # First result should be vec1 (closest to query)
            assert data["results"][0]["id"] == "vec1"

    def test_delete_vector(self, api_client, test_tenant):
        """Test deleting a vector from an index"""
        # First add vectors to an index
        index_id = self.test_add_vectors(api_client, test_tenant)

        # Delete one vector
        response = api_client.delete(f"/v1/index/{index_id}/vectors/vec1")
        print(f"Delete vector response status: {response.status_code}")
        if response.status_code != status.HTTP_204_NO_CONTENT:
            print(f"Error response: {response.text}")

        # Check response
        assert response.status_code == status.HTTP_204_NO_CONTENT

        # Verify the vector is gone by searching
        search_data = {
            "vector": [1.0, 0.0, 0.0, 0.0],  # Exactly vec1
            "k": 1
        }

        # Use POST for search instead of GET since we need to send a JSON body
        search_response = api_client.post(f"/v1/index/{index_id}/search", json=search_data)

        # Should still return a valid result, but not vec1
        assert search_response.status_code == status.HTTP_200_OK
        data = search_response.json()

        # Check if results exist
        if data["results"] and len(data["results"]) > 0:
            # The result should not be vec1
            assert data["results"][0]["id"] != "vec1"

    def test_add_vectors_nonexistent_index(self, api_client):
        """Test adding vectors to a nonexistent index"""
        # Prepare vector data
        vectors_data = {
            "vectors": [
                {
                    "id": "vec1",
                    "values": [1.0, 0.0, 0.0, 0.0],
                    "metadata": {}
                }
            ]
        }

        # Try to add vectors to a nonexistent index
        response = api_client.post("/v1/index/nonexistent-index/vectors", json=vectors_data)
        print(f"Add to nonexistent index response status: {response.status_code}")
        if response.status_code != status.HTTP_404_NOT_FOUND:
            print(f"Error response: {response.json()}")

        # Check response - should be not found
        assert response.status_code == status.HTTP_404_NOT_FOUND
