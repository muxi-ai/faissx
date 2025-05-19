#!/usr/bin/env python3
"""
FAISS Proxy Server Example

This example shows how to configure and run the FAISS Proxy server.
"""

from faiss_proxy import server

# Configure the server with default in-memory storage
server.configure(
    port=45678,
    bind_address="0.0.0.0",
    # data_dir is omitted, so it will use in-memory indices

    # Method 1: Directly specify API keys
    auth_keys={"test-key-1": "tenant-1", "test-key-2": "tenant-2"},
    enable_auth=True

    # Method 2: Load API keys from a JSON file (can't use both methods)
    # auth_file="examples/auth.json",
    # enable_auth=True
)

print("Starting FAISS Proxy Server...")
print(f"Configuration: {server.get_config()}")

# To use a specific data directory instead, you would configure like this:
# server.configure(
#     port=45678,
#     bind_address="0.0.0.0",
#     data_dir="./data",  # Specify a directory for persistence
#     auth_keys={"test-key-1": "tenant-1", "test-key-2": "tenant-2"},
#     enable_auth=True
# )

# Run the server (this will block until the server is stopped)
server.run()
