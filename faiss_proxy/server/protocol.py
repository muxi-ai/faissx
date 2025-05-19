import msgpack
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

# Constants for message structure
HEADER_FIELDS = ["operation", "api_key", "tenant_id", "index_id", "request_id"]
VECTOR_DTYPE = np.float32


class ProtocolError(Exception):
    """Protocol parsing or formatting error"""

    pass


def serialize_message(
    header: Dict[str, Any],
    vectors: Optional[np.ndarray] = None,
    metadata: Optional[Any] = None,
) -> bytes:
    """
    Serialize a message for sending over ZeroMQ.

    Args:
        header: Message header with operation, auth, etc.
        vectors: Optional numpy array of vectors
        metadata: Optional metadata to include

    Returns:
        bytes: Serialized message
    """
    # 1. Serialize header with msgpack
    header_bytes = msgpack.packb(header)
    header_size = len(header_bytes)

    # 2. Prepare parts
    parts = [header_bytes]

    # 3. Add vector data if present
    vector_size = 0
    if vectors is not None:
        # Ensure vectors are float32
        if vectors.dtype != VECTOR_DTYPE:
            vectors = vectors.astype(VECTOR_DTYPE)
        vector_bytes = vectors.tobytes()
        vector_size = len(vector_bytes)
        parts.append(vector_bytes)

    # 4. Add metadata if present
    metadata_size = 0
    if metadata is not None:
        metadata_bytes = msgpack.packb(metadata)
        metadata_size = len(metadata_bytes)
        parts.append(metadata_bytes)

    # 5. Create sizes header
    sizes = msgpack.packb(
        {
            "header_size": header_size,
            "vector_size": vector_size,
            "metadata_size": metadata_size,
        }
    )

    # 6. Combine all parts
    return sizes + b"".join(parts)


def deserialize_message(
    data: bytes,
) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[Any]]:
    """
    Deserialize a message received over ZeroMQ.

    Args:
        data: Raw message bytes

    Returns:
        Tuple containing:
        - header: Dictionary with message header
        - vectors: Numpy array of vectors (if present)
        - metadata: Metadata object (if present)

    Raises:
        ProtocolError: If message format is invalid
    """
    try:
        # 1. Extract sizes header
        sizes_end = data.find(b"\xc0")  # Find end of msgpack map
        if sizes_end == -1:
            raise ProtocolError("Invalid message format: can't find sizes header")

        sizes = msgpack.unpackb(data[: sizes_end + 1])

        # 2. Extract parts based on sizes
        header_size = sizes.get("header_size", 0)
        vector_size = sizes.get("vector_size", 0)
        metadata_size = sizes.get("metadata_size", 0)

        offset = sizes_end + 1

        # 3. Extract header
        header = msgpack.unpackb(data[offset : offset + header_size])
        offset += header_size

        # 4. Extract vectors if present
        vectors = None
        if vector_size > 0:
            vector_data = data[offset : offset + vector_size]

            # We need shape information to reconstruct the array
            # This should be part of the header for search/add operations
            if "vector_shape" in header:
                shape = header["vector_shape"]
                vectors = np.frombuffer(vector_data, dtype=VECTOR_DTYPE).reshape(shape)
            else:
                # For single vector queries
                dimension = header.get("dimension", 0)
                if dimension > 0:
                    count = vector_size // (dimension * 4)  # 4 bytes per float32
                    vectors = np.frombuffer(vector_data, dtype=VECTOR_DTYPE).reshape(
                        count, dimension
                    )
                else:
                    # Just return the raw buffer if we can't determine shape
                    vectors = np.frombuffer(vector_data, dtype=VECTOR_DTYPE)

            offset += vector_size

        # 5. Extract metadata if present
        metadata = None
        if metadata_size > 0:
            metadata = msgpack.unpackb(data[offset : offset + metadata_size])

        return header, vectors, metadata

    except (msgpack.UnpackException, IndexError, ValueError) as e:
        raise ProtocolError(f"Failed to deserialize message: {str(e)}")


# --- Operation-specific serialization/deserialization ---


def prepare_create_index_request(
    api_key: str,
    tenant_id: str,
    name: str,
    dimension: int,
    index_type: str = "IndexFlatL2",
) -> bytes:
    """
    Prepare a create_index request message.

    Args:
        api_key: API key for authentication
        tenant_id: Tenant ID
        name: Index name
        dimension: Vector dimension
        index_type: FAISS index type

    Returns:
        bytes: Serialized message
    """
    header = {
        "operation": "create_index",
        "api_key": api_key,
        "tenant_id": tenant_id,
        "name": name,
        "dimension": dimension,
        "index_type": index_type,
    }
    return serialize_message(header)


def prepare_add_vectors_request(
    api_key: str,
    tenant_id: str,
    index_id: str,
    vectors: np.ndarray,
    vector_ids: List[str],
    vector_metadata: List[Dict[str, Any]],
) -> bytes:
    """
    Prepare an add_vectors request message.

    Args:
        api_key: API key for authentication
        tenant_id: Tenant ID
        index_id: Index ID
        vectors: Numpy array of vectors to add (shape: N x D)
        vector_ids: List of vector IDs
        vector_metadata: List of vector metadata dicts

    Returns:
        bytes: Serialized message
    """
    header = {
        "operation": "add_vectors",
        "api_key": api_key,
        "tenant_id": tenant_id,
        "index_id": index_id,
        "vector_shape": vectors.shape,
    }

    # Package metadata with IDs
    metadata = []
    for i, vector_id in enumerate(vector_ids):
        metadata.append(
            {
                "id": vector_id,
                "metadata": vector_metadata[i] if i < len(vector_metadata) else {},
            }
        )

    return serialize_message(header, vectors, metadata)


def prepare_search_request(
    api_key: str,
    tenant_id: str,
    index_id: str,
    query_vector: np.ndarray,
    k: int = 10,
    filter_metadata: Optional[Dict[str, Any]] = None,
) -> bytes:
    """
    Prepare a search request message.

    Args:
        api_key: API key for authentication
        tenant_id: Tenant ID
        index_id: Index ID
        query_vector: Query vector (shape: D)
        k: Number of results to return
        filter_metadata: Optional metadata filter

    Returns:
        bytes: Serialized message
    """
    header = {
        "operation": "search",
        "api_key": api_key,
        "tenant_id": tenant_id,
        "index_id": index_id,
        "k": k,
        "dimension": query_vector.shape[0],
    }

    # Ensure query vector is correctly shaped (D) -> (1, D)
    if len(query_vector.shape) == 1:
        query_vector = query_vector.reshape(1, -1)

    return serialize_message(header, query_vector, filter_metadata)


def prepare_delete_vector_request(
    api_key: str, tenant_id: str, index_id: str, vector_id: str
) -> bytes:
    """
    Prepare a delete_vector request message.

    Args:
        api_key: API key for authentication
        tenant_id: Tenant ID
        index_id: Index ID
        vector_id: Vector ID to delete

    Returns:
        bytes: Serialized message
    """
    header = {
        "operation": "delete_vector",
        "api_key": api_key,
        "tenant_id": tenant_id,
        "index_id": index_id,
        "vector_id": vector_id,
    }
    return serialize_message(header)


def prepare_get_index_info_request(
    api_key: str, tenant_id: str, index_id: str
) -> bytes:
    """
    Prepare a get_index_info request message.

    Args:
        api_key: API key for authentication
        tenant_id: Tenant ID
        index_id: Index ID

    Returns:
        bytes: Serialized message
    """
    header = {
        "operation": "get_index_info",
        "api_key": api_key,
        "tenant_id": tenant_id,
        "index_id": index_id,
    }
    return serialize_message(header)


def prepare_delete_index_request(api_key: str, tenant_id: str, index_id: str) -> bytes:
    """
    Prepare a delete_index request message.

    Args:
        api_key: API key for authentication
        tenant_id: Tenant ID
        index_id: Index ID

    Returns:
        bytes: Serialized message
    """
    header = {
        "operation": "delete_index",
        "api_key": api_key,
        "tenant_id": tenant_id,
        "index_id": index_id,
    }
    return serialize_message(header)


# --- Response formatting ---


def prepare_success_response(result: Any = None) -> bytes:
    """
    Prepare a success response message.

    Args:
        result: Result data to include in the response

    Returns:
        bytes: Serialized message
    """
    header = {"status": "ok"}
    return serialize_message(header, None, result)


def prepare_error_response(
    error_type: str, message: str, request_id: Optional[str] = None
) -> bytes:
    """
    Prepare an error response message.

    Args:
        error_type: Type of error
        message: Error message
        request_id: Optional request ID for tracing

    Returns:
        bytes: Serialized message
    """
    header = {"status": "error", "error_type": error_type, "message": message}
    if request_id:
        header["request_id"] = request_id

    return serialize_message(header)
