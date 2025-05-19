#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Unified interface for LLM providers using OpenAI format
# https://github.com/muxi-ai/faissx
#
# Copyright (C) 2025 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FAISSx Server Example

This example shows how to configure and run the FAISSx server.
"""

from faissx import server

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

print("Starting FAISSx Server...")
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
