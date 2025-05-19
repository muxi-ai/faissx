#!/usr/bin/env python3
"""
FAISS Proxy Server - ZeroMQ Implementation
A standalone server that provides vector operations via ZeroMQ
"""

import os
import time
import zmq
import numpy as np
import faiss
import msgpack

# Constants
DEFAULT_PORT = 45678
DEFAULT_BIND_ADDRESS = "0.0.0.0"


class FaissIndex:
    """Manages FAISS indexes for vector storage and search"""

    def __init__(self, data_dir=None):
        """
        Initialize the FAISS index manager.

        Args:
            data_dir: Directory to store FAISS indices. If None, uses in-memory indices
                     without persistence.
        """
        self.data_dir = data_dir
        self.indexes = {}
        self.dimensions = {}

        if data_dir:
            os.makedirs(data_dir, exist_ok=True)
            print(f"FAISS Index initialized with data directory: {data_dir}")
        else:
            print("FAISS Index initialized with in-memory indices (no persistence)")

    def create_index(self, index_id, dimension, index_type="L2"):
        """Create a new FAISS index"""
        if index_id in self.indexes:
            return {"success": False, "error": f"Index {index_id} already exists"}

        try:
            if index_type == "L2":
                index = faiss.IndexFlatL2(dimension)
            elif index_type == "IP":
                index = faiss.IndexFlatIP(dimension)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported index type: {index_type}",
                }

            self.indexes[index_id] = index
            self.dimensions[index_id] = dimension
            print(
                f"Created index {index_id} with dimension {dimension}, type {index_type}"
            )
            return {
                "success": True,
                "index_id": index_id,
                "dimension": dimension,
                "type": index_type,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def add_vectors(self, index_id, vectors):
        """Add vectors to an existing index"""
        if index_id not in self.indexes:
            return {"success": False, "error": f"Index {index_id} does not exist"}

        try:
            vectors_np = np.array(vectors, dtype=np.float32)
            if vectors_np.shape[1] != self.dimensions[index_id]:
                return {
                    "success": False,
                    "error": f"Vector dimension mismatch. Expected {self.dimensions[index_id]}, got {vectors_np.shape[1]}",
                }

            self.indexes[index_id].add(vectors_np)
            total = self.indexes[index_id].ntotal
            print(f"Added {len(vectors)} vectors to index {index_id}, total: {total}")
            return {"success": True, "count": len(vectors), "total": total}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def search(self, index_id, query_vectors, k=10):
        """Search for similar vectors in an index"""
        if index_id not in self.indexes:
            return {"success": False, "error": f"Index {index_id} does not exist"}

        try:
            query_np = np.array(query_vectors, dtype=np.float32)
            if query_np.shape[1] != self.dimensions[index_id]:
                return {
                    "success": False,
                    "error": f"Query vector dimension mismatch. Expected {self.dimensions[index_id]}, got {query_np.shape[1]}",
                }

            distances, indices = self.indexes[index_id].search(query_np, k)

            # Convert to Python lists for serialization
            results = []
            for i in range(len(query_vectors)):
                results.append(
                    {"distances": distances[i].tolist(), "indices": indices[i].tolist()}
                )

            print(f"Searched index {index_id} with {len(query_vectors)} queries, k={k}")
            return {"success": True, "results": results}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_index_stats(self, index_id):
        """Get statistics for an index"""
        if index_id not in self.indexes:
            return {"success": False, "error": f"Index {index_id} does not exist"}

        try:
            index = self.indexes[index_id]
            stats = {
                "index_id": index_id,
                "dimension": self.dimensions[index_id],
                "vector_count": index.ntotal,
                "type": "L2" if isinstance(index, faiss.IndexFlatL2) else "IP",
            }
            return {"success": True, "stats": stats}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_indexes(self):
        """List all available indexes"""
        try:
            index_list = []
            for index_id in self.indexes:
                index_list.append(
                    {
                        "index_id": index_id,
                        "dimension": self.dimensions[index_id],
                        "vector_count": self.indexes[index_id].ntotal,
                    }
                )
            return {"success": True, "indexes": index_list}
        except Exception as e:
            return {"success": False, "error": str(e)}


def serialize_message(data):
    """Serialize a message to binary format"""
    if isinstance(data, dict) and "results" in data and data.get("success", False):
        # Special handling for search results with numpy arrays
        return msgpack.packb(data, use_bin_type=True)
    else:
        # Regular JSON-serializable data
        return msgpack.packb(data, use_bin_type=True)


def deserialize_message(data):
    """Deserialize a binary message"""
    try:
        return msgpack.unpackb(data, raw=False)
    except Exception as e:
        return {"success": False, "error": f"Failed to deserialize message: {str(e)}"}


def authenticate_request(request, auth_keys):
    """Authenticate a request using API keys"""
    if not auth_keys:
        # Authentication disabled
        return True, None

    api_key = request.get("api_key")
    if not api_key:
        return False, "API key required"

    tenant_id = auth_keys.get(api_key)
    if not tenant_id:
        return False, "Invalid API key"

    # Add tenant_id to the request
    request["tenant_id"] = tenant_id
    return True, None


def run_server(
    port=DEFAULT_PORT,
    bind_address=DEFAULT_BIND_ADDRESS,
    auth_keys=None,
    enable_auth=False,
    data_dir=None,
):
    """Run the ZeroMQ server"""
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{bind_address}:{port}")

    faiss_index = FaissIndex(data_dir=data_dir)

    print(f"FAISS Proxy Server (ZeroMQ) started on {bind_address}:{port}")
    print(f"Using FAISS version: {faiss.__version__}")
    if data_dir:
        print(f"Data directory: {data_dir}")
    else:
        print("Using in-memory indices (no persistence)")
    print(f"Authentication enabled: {enable_auth}")

    while True:
        try:
            # Wait for next request from client
            message = socket.recv()
            try:
                request = deserialize_message(message)
                action = request.get("action", "")

                # Handle authentication if enabled
                if enable_auth:
                    is_authenticated, error = authenticate_request(request, auth_keys)
                    if not is_authenticated:
                        response = {"success": False, "error": error}
                        socket.send(serialize_message(response))
                        continue

                if action == "create_index":
                    response = faiss_index.create_index(
                        request.get("index_id", ""),
                        request.get("dimension", 0),
                        request.get("index_type", "L2"),
                    )

                elif action == "add_vectors":
                    response = faiss_index.add_vectors(
                        request.get("index_id", ""), request.get("vectors", [])
                    )

                elif action == "search":
                    response = faiss_index.search(
                        request.get("index_id", ""),
                        request.get("query_vectors", []),
                        request.get("k", 10),
                    )

                elif action == "get_index_stats":
                    response = faiss_index.get_index_stats(request.get("index_id", ""))

                elif action == "list_indexes":
                    response = faiss_index.list_indexes()

                elif action == "ping":
                    response = {"success": True, "message": "pong", "time": time.time()}

                else:
                    response = {"success": False, "error": f"Unknown action: {action}"}

            except Exception as e:
                response = {"success": False, "error": str(e)}

            # Send reply back to client
            socket.send(serialize_message(response))

        except KeyboardInterrupt:
            print("Server shutting down...")
            break
        except Exception as e:
            print(f"Error: {e}")

    socket.close()
    context.term()


if __name__ == "__main__":
    port = int(os.environ.get("FAISS_PROXY_PORT", DEFAULT_PORT))
    data_dir = os.environ.get("FAISS_DATA_DIR", None)
    bind_address = os.environ.get("FAISS_BIND_ADDRESS", DEFAULT_BIND_ADDRESS)

    # Default to no authentication when run directly
    run_server(port, bind_address, auth_keys={}, enable_auth=False, data_dir=data_dir)
