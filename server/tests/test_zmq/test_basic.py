import pytest
import numpy as np
import uuid
import time
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from protocol import (
    prepare_create_index_request,
    prepare_add_vectors_request,
    prepare_search_request,
    prepare_get_index_info_request,
    prepare_delete_vector_request,
    prepare_delete_index_request,
    deserialize_message
)


def test_create_index(zmq_client, test_api_key, test_tenant):
    """Test creating an index"""
    dimension = 128
    name = f"test-index-{uuid.uuid4()}"

    # Prepare request
    request = prepare_create_index_request(
        api_key=test_api_key,
        tenant_id=test_tenant,
        name=name,
        dimension=dimension
    )

    # Send request
    zmq_client.send(request)

    # Receive response
    response = zmq_client.recv()
    header, _, _ = deserialize_message(response)

    # Verify response
    assert header.get("status") == "ok"
    assert "result" in header
    assert "index_id" in header["result"]

    index_id = header["result"]["index_id"]
    return index_id


def test_add_vectors(zmq_client, test_api_key, test_tenant):
    """Test adding vectors to an index"""
    # First create an index
    index_id = test_create_index(zmq_client, test_api_key, test_tenant)

    # Generate test vectors
    num_vectors = 10
    dimension = 128
    vectors = np.random.random((num_vectors, dimension)).astype(np.float32)
    vector_ids = [f"vector-{i}" for i in range(num_vectors)]
    vector_metadata = [{"value": i, "timestamp": time.time()} for i in range(num_vectors)]

    # Prepare request
    request = prepare_add_vectors_request(
        api_key=test_api_key,
        tenant_id=test_tenant,
        index_id=index_id,
        vectors=vectors,
        vector_ids=vector_ids,
        vector_metadata=vector_metadata
    )

    # Send request
    zmq_client.send(request)

    # Receive response
    response = zmq_client.recv()
    header, _, _ = deserialize_message(response)

    # Verify response
    assert header.get("status") == "ok"
    assert "result" in header
    assert header["result"].get("added_count", 0) == num_vectors

    return index_id, vectors, vector_ids


def test_search(zmq_client, test_api_key, test_tenant):
    """Test searching for vectors"""
    # First add vectors to an index
    index_id, vectors, vector_ids = test_add_vectors(zmq_client, test_api_key, test_tenant)

    # Use first vector as query
    query_vector = vectors[0]
    k = 3

    # Prepare request
    request = prepare_search_request(
        api_key=test_api_key,
        tenant_id=test_tenant,
        index_id=index_id,
        query_vector=query_vector,
        k=k
    )

    # Send request
    zmq_client.send(request)

    # Receive response
    response = zmq_client.recv()
    header, _, _ = deserialize_message(response)

    # Verify response
    assert header.get("status") == "ok"
    assert "result" in header

    results = header["result"]
    assert isinstance(results, list)
    assert len(results) > 0
    assert len(results) <= k

    # The top result should be the query vector itself with high similarity
    assert results[0]["id"] == vector_ids[0]
    assert results[0]["score"] > 0.9  # Score should be close to 1.0 for identical vector


def test_get_index_info(zmq_client, test_api_key, test_tenant):
    """Test getting index information"""
    # First create an index
    index_id = test_create_index(zmq_client, test_api_key, test_tenant)

    # Prepare request
    request = prepare_get_index_info_request(
        api_key=test_api_key,
        tenant_id=test_tenant,
        index_id=index_id
    )

    # Send request
    zmq_client.send(request)

    # Receive response
    response = zmq_client.recv()
    header, _, _ = deserialize_message(response)

    # Verify response
    assert header.get("status") == "ok"
    assert "result" in header

    info = header["result"]
    assert info["id"] == index_id
    assert "dimension" in info
    assert "vector_count" in info
    assert info["tenant_id"] == test_tenant


def test_delete_vector(zmq_client, test_api_key, test_tenant):
    """Test deleting a vector"""
    # First add vectors to an index
    index_id, _, vector_ids = test_add_vectors(zmq_client, test_api_key, test_tenant)

    # Delete the first vector
    vector_id = vector_ids[0]

    # Prepare request
    request = prepare_delete_vector_request(
        api_key=test_api_key,
        tenant_id=test_tenant,
        index_id=index_id,
        vector_id=vector_id
    )

    # Send request
    zmq_client.send(request)

    # Receive response
    response = zmq_client.recv()
    header, _, _ = deserialize_message(response)

    # Verify response
    assert header.get("status") == "ok"
    assert "result" in header
    assert header["result"].get("deleted") is True


def test_delete_index(zmq_client, test_api_key, test_tenant):
    """Test deleting an index"""
    # First create an index
    index_id = test_create_index(zmq_client, test_api_key, test_tenant)

    # Prepare request
    request = prepare_delete_index_request(
        api_key=test_api_key,
        tenant_id=test_tenant,
        index_id=index_id
    )

    # Send request
    zmq_client.send(request)

    # Receive response
    response = zmq_client.recv()
    header, _, _ = deserialize_message(response)

    # Verify response
    assert header.get("status") == "ok"
    assert "result" in header
    assert header["result"].get("deleted") is True

    # Verify that getting the index now fails
    request = prepare_get_index_info_request(
        api_key=test_api_key,
        tenant_id=test_tenant,
        index_id=index_id
    )

    zmq_client.send(request)
    response = zmq_client.recv()
    header, _, _ = deserialize_message(response)

    assert header.get("status") == "error"
