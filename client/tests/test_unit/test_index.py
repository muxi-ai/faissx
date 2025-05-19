import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from faissx.index import IndexFlatL2


class TestIndexFlatL2:
    """Unit tests for the IndexFlatL2 class"""

    @patch("faissx.index.get_client")
    def test_index_init(self, mock_get_client):
        """Test initialization of the IndexFlatL2 class"""
        # Setup mock
        mock_client = MagicMock()
        mock_client.create_index.return_value = "test-index-id"
        mock_get_client.return_value = mock_client

        # Create index
        index = IndexFlatL2(128)

        # Check instance attributes
        assert index.d == 128
        assert index.is_trained is True
        assert index.ntotal == 0
        assert index.index_id == "test-index-id"
        assert index._next_id == 0
        assert index._vector_ids == []

        # Check that client methods were called correctly
        mock_get_client.assert_called_once()
        mock_client.create_index.assert_called_once()
        args, kwargs = mock_client.create_index.call_args
        assert kwargs["dimension"] == 128
        assert kwargs["index_type"] == "IndexFlatL2"
        assert "name" in kwargs and kwargs["name"].startswith("index-flat-l2-")

    @patch("faissx.index.get_client")
    def test_add_vectors(self, mock_get_client):
        """Test adding vectors to the index"""
        # Setup mock
        mock_client = MagicMock()
        mock_client.create_index.return_value = "test-index-id"
        mock_client.add_vectors.return_value = {
            "success": True,
            "added_count": 2,
            "failed_count": 0
        }
        mock_get_client.return_value = mock_client

        # Create index and add vectors
        index = IndexFlatL2(3)
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])

        index.add(vectors)

        # Check that vectors were added
        assert index.ntotal == 2
        assert len(index._vector_ids) == 2
        assert index._next_id == 2

        # Check that client methods were called correctly
        mock_client.add_vectors.assert_called_once()
        args, kwargs = mock_client.add_vectors.call_args
        assert args[0] == "test-index-id"  # First arg is index_id

        # Second arg is the vectors dict
        vectors_dict = args[1]
        assert "vectors" in vectors_dict
        assert len(vectors_dict["vectors"]) == 2

        # Check vector values
        vector_values = [v["values"] for v in vectors_dict["vectors"]]
        assert [1.0, 0.0, 0.0] in vector_values
        assert [0.0, 1.0, 0.0] in vector_values

    @patch("faissx.index.get_client")
    def test_search(self, mock_get_client):
        """Test searching for similar vectors"""
        # Setup mock
        mock_client = MagicMock()
        mock_client.create_index.return_value = "test-index-id"
        mock_client.search.return_value = {
            "results": [
                {"id": "vec-0", "score": 0.9, "metadata": {}},
                {"id": "vec-1", "score": 0.5, "metadata": {}}
            ]
        }
        mock_get_client.return_value = mock_client

        # Create index with some vectors
        index = IndexFlatL2(3)

        # Add vectors to setup the vector ID mapping
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        # Mock the add to avoid actually calling the API
        mock_client.add_vectors.return_value = {
            "success": True,
            "added_count": 2,
            "failed_count": 0
        }
        index.add(vectors)

        # Perform search
        query = np.array([[0.9, 0.1, 0.0]])
        D, I = index.search(query, k=2)

        # Check results
        assert D.shape == (1, 2)
        assert I.shape == (1, 2)

        # First result should be vec-0 with index 0
        assert I[0, 0] == 0
        # Second result should be vec-1 with index 1
        assert I[0, 1] == 1

        # Check distances (higher score -> lower distance)
        assert D[0, 0] < D[0, 1]  # First should have lower distance

        # Check that client methods were called correctly
        mock_client.search.assert_called_once()
        args, kwargs = mock_client.search.call_args
        assert args[0] == "test-index-id"
        assert kwargs["vector"] == [0.9, 0.1, 0.0]
        assert kwargs["k"] == 2

    @patch("faissx.index.get_client")
    def test_reset(self, mock_get_client):
        """Test resetting the index"""
        # Setup mock
        mock_client = MagicMock()
        mock_client.create_index.return_value = "test-index-id"
        mock_get_client.return_value = mock_client

        # Create index and add vectors
        index = IndexFlatL2(3)
        index.ntotal = 10  # Simulate adding vectors
        index._next_id = 10
        index._vector_ids = ["vec-0", "vec-1", "vec-2"]

        # Reset index
        index.reset()

        # Check that the index was reset
        assert index.ntotal == 0
        assert index._next_id == 0
        assert index._vector_ids == []

        # Check that client methods were called correctly
        mock_client.delete_index.assert_called_once_with("test-index-id")
        assert mock_client.create_index.call_count == 2  # Once for init, once for reset

    @patch("faissx.index.get_client")
    def test_invalid_vector_shape(self, mock_get_client):
        """Test handling of invalid vector shapes"""
        # Setup mock
        mock_client = MagicMock()
        mock_client.create_index.return_value = "test-index-id"
        mock_get_client.return_value = mock_client

        # Create index
        index = IndexFlatL2(3)

        # Try to add vectors with wrong dimension
        with pytest.raises(ValueError, match="Invalid vector shape"):
            index.add(np.array([[1.0, 0.0]]))  # Should be (n, 3)

        # Try to search with wrong dimension
        with pytest.raises(ValueError, match="Invalid vector shape"):
            index.search(np.array([[1.0, 0.0]]), k=1)  # Should be (n, 3)
