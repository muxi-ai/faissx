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
FAISSx Client Example (Placeholder)

This example shows how the client will be used once implemented.
Note: This is just a placeholder for the future client implementation.
"""

import numpy as np
import faissx as faiss

# Configure the client
faiss.configure(
    server="tcp://localhost:45678",
    api_key="test-key-1",
    tenant_id="tenant-1"
)

# Create an index (example of future API)
dimension = 128
index = faiss.IndexFlatL2(dimension)  # This will be implemented later

# Create some random vectors
num_vectors = 100
vectors = np.random.random((num_vectors, dimension)).astype('float32')

# Add vectors (example of future API)
index.add(vectors)  # This will be implemented later

# Search for similar vectors
query = np.random.random((1, dimension)).astype('float32')

# Perform search (example of future API)
k = 5
D, I = index.search(query, k)  # This will be implemented later

print(f"Found {len(I[0])} matches")
print(f"Distances: {D[0]}")
print(f"Indices: {I[0]}")

print("\nNOTE: This is just a placeholder. The actual client implementation will be developed later.")
