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
Binary Hash Index implementation for FAISSx.

This module provides a binary vector index using hash-based lookup with Hamming distance,
which is compatible with the FAISS IndexBinaryHash.
"""

import numpy as np
from typing import Dict, List, Tuple, Set

from faissx.client.indices.binary_base import BinaryIndex
from faissx.client.client import get_client


class IndexBinaryHash(BinaryIndex):
    """
    IndexBinaryHash is a binary vector index that uses hash-based lookup with Hamming distance.

    This index uses bit sampling and hash tables to speed up search by only comparing
    vectors that have similar hash codes. It is compatible with the FAISS IndexBinaryHash
    implementation.
    """

    def __init__(self, d: int, nhash: int = 4, hash_bits: int = 16):
        """
        Initialize the binary hash index.

        Args:
            d (int): Dimension of binary vectors in bytes.
                    (Each byte contains 8 bits, so the actual dimension in bits is 8*d)
            nhash (int, optional): Number of hash tables to use. Defaults to 4.
            hash_bits (int, optional): Number of bits to use for hash keys. Defaults to 16.
        """
        super().__init__(d)  # Initialize binary base class

        # Store parameters
        self.nhash: int = nhash
        self.hash_bits: int = hash_bits
        self.is_trained: bool = False  # Hash indices need training for bit selection

        # Hash table parameters
        self.hash_tables: List[Dict[int, List[int]]] = []
        self.bit_masks: List[np.ndarray] = []
        self.bucket_threshold: int = 1000  # Maximum bucket size for full search

    def _select_bits(self, vectors: np.ndarray) -> None:
        """
        Select bits to use for hashing.

        This is a simplified implementation that randomly selects bits.

        Args:
            vectors (np.ndarray): Training vectors, uint8 array of shape (n, d)
        """
        total_bits = self.d * 8
        bits_per_hash = self.hash_bits

        # Generate random bit selections for each hash table
        for _ in range(self.nhash):
            # Random bit sampling
            bit_indices = np.random.choice(total_bits, bits_per_hash, replace=False)

            # Convert bit indices to byte indices and bit positions
            byte_indices = bit_indices // 8
            bit_positions = bit_indices % 8

            # Create a mask for each bit position
            mask = np.zeros((bits_per_hash, self.d), dtype=np.uint8)
            for i, (byte_idx, bit_pos) in enumerate(zip(byte_indices, bit_positions)):
                mask[i, byte_idx] = 1 << (7 - bit_pos)  # Bit mask (MSB first)

            self.bit_masks.append(mask)

    def _compute_hash(self, vector: np.ndarray, mask: np.ndarray) -> int:
        """
        Compute a hash code for a vector using a bit mask.

        Args:
            vector (np.ndarray): Binary vector, uint8 array of shape (d,)
            mask (np.ndarray): Bit mask of shape (hash_bits, d)

        Returns:
            int: Hash code
        """
        hash_code = 0
        for i, byte_mask in enumerate(mask):
            # Apply mask to vector
            masked = vector & byte_mask

            # Count bits set in masked result
            bit_count = 0
            for byte in masked:
                bit_count += bin(byte).count("1")

            # Set bit in hash code if at least one bit is set
            if bit_count > 0:
                hash_code |= (1 << i)

        return hash_code

    def train(self, vectors: np.ndarray) -> None:
        """
        Train the index by selecting bits for hashing.

        Args:
            vectors (np.ndarray): Training vectors, uint8 array of shape (n, d)
        """
        # Validate input
        self.validate_binary_vectors(vectors, operation="train")

        # Get client for potential remote operations
        client = get_client()

        if client and client.is_remote_mode():
            # Remote mode: send to server
            # Placeholder for server implementation - to be completed
            client.train(self.index_id, vectors)
        else:
            # Local mode: select bits for hashing
            self._select_bits(vectors)

            # Initialize hash tables
            self.hash_tables = [{} for _ in range(self.nhash)]

        self.is_trained = True

    def add(self, vectors: np.ndarray) -> None:
        """
        Add binary vectors to the index.

        Args:
            vectors (np.ndarray): Binary vectors to add, should be uint8 with shape
                                  (n, d) where d is the dimension in bytes
        """
        # Validate input
        self.validate_binary_vectors(vectors)

        # Check if trained
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors")

        # Determine starting index for new vectors
        start_idx = self.ntotal
        n = vectors.shape[0]

        # Get client for potential remote operations
        client = get_client()

        if client and client.is_remote_mode():
            # Remote mode: send to server
            # Placeholder for server implementation - to be completed
            client.add_vectors(self.index_id, vectors)

            # Update vector mapping for remote mode
            for i in range(n):
                self._vector_mapping[start_idx + i] = {"server_id": start_idx + i}
        else:
            # Local mode: store vectors and add to hash tables
            if self._vectors is None:
                self._vectors = vectors.copy()
            else:
                self._vectors = np.vstack((self._vectors, vectors))

            # Add vectors to hash tables
            for i in range(n):
                vector_idx = start_idx + i
                vector = vectors[i]

                # Compute hash codes for each table and add to buckets
                for table_idx, (hash_table, bit_mask) in enumerate(zip(self.hash_tables, self.bit_masks)):
                    hash_code = self._compute_hash(vector, bit_mask)

                    # Add vector index to hash bucket
                    if hash_code not in hash_table:
                        hash_table[hash_code] = []
                    hash_table[hash_code].append(vector_idx)

                # Update vector mapping for local mode
                self._vector_mapping[vector_idx] = {"local_id": vector_idx}

        # Update total count
        self.ntotal += n

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors using Hamming distance with hash-based lookup.

        Args:
            query (np.ndarray): Binary query vectors, uint8 with shape (n, d)
            k (int): Number of nearest neighbors to return

        Returns:
            Tuple[np.ndarray, np.ndarray]: Distances and indices of nearest neighbors
                - distances: Hamming distances with shape (n, k)
                - indices: Vector indices with shape (n, k)
        """
        # Validate input
        self.validate_binary_vectors(query, operation="search")

        # Check if trained
        if not self.is_trained:
            raise RuntimeError("Index must be trained before searching")

        # Get client for potential remote operations
        client = get_client()

        if client and client.is_remote_mode():
            # Remote mode: send to server
            # Placeholder for server implementation - to be completed
            distances, indices = client.search(self.index_id, query, k)
            return distances, indices
        else:
            # Local mode: search using hash tables
            if self.ntotal == 0 or self._vectors is None:
                # Return empty results if no vectors
                n = query.shape[0]
                return np.zeros((n, k), dtype=np.float32), np.zeros((n, k), dtype=np.int64)

            n_queries = query.shape[0]

            # Initialize results
            results_distances = np.full((n_queries, k), np.inf, dtype=np.float32)
            results_indices = np.zeros((n_queries, k), dtype=np.int64)

            for i in range(n_queries):
                # Collect candidates from hash tables
                candidates: Set[int] = set()
                q = query[i]

                for table_idx, (hash_table, bit_mask) in enumerate(zip(self.hash_tables, self.bit_masks)):
                    hash_code = self._compute_hash(q, bit_mask)

                    # Add vectors in the same bucket as candidates
                    if hash_code in hash_table:
                        candidates.update(hash_table[hash_code])

                if not candidates:
                    continue  # No candidates found

                # Convert set to list for indexing
                candidate_list = list(candidates)

                # If too many candidates, fall back to full search
                if len(candidate_list) > self.bucket_threshold:
                    candidate_list = list(range(self.ntotal))

                # Compute distances to all candidates
                candidate_vectors = self._vectors[candidate_list]
                # Get Hamming distances
                distances_result = self.compute_hamming_distances(query[i:i+1], candidate_vectors)
                candidate_distances = distances_result[0]

                # Get top k results
                if len(candidate_list) <= k:
                    # If fewer candidates than k, return all
                    top_k = np.argsort(candidate_distances)
                    results_distances[i, :len(candidate_list)] = candidate_distances[top_k]
                    results_indices[i, :len(candidate_list)] = [candidate_list[j] for j in top_k]
                else:
                    # Get top k
                    top_k = np.argpartition(candidate_distances, k)[:k]
                    # Sort the top k
                    sorted_indices = np.argsort(candidate_distances[top_k])
                    sorted_top_k = top_k[sorted_indices]
                    results_distances[i, :k] = candidate_distances[sorted_top_k]
                    results_indices[i, :k] = [candidate_list[j] for j in sorted_top_k]

            return results_distances, results_indices

    def reconstruct(self, idx: int) -> np.ndarray:
        """
        Reconstruct the binary vector at the given index.

        Args:
            idx (int): Index of the vector to reconstruct

        Returns:
            np.ndarray: The reconstructed binary vector
        """
        if idx < 0 or idx >= self.ntotal:
            raise IndexError(f"Index {idx} out of bounds for index with {self.ntotal} vectors")

        # Get client for potential remote operations
        client = get_client()

        if client and client.is_remote_mode():
            # Remote mode: request from server
            # Placeholder for server implementation - to be completed
            return client.reconstruct(self.index_id, idx)
        else:
            # Local mode: return stored vector
            if self._vectors is None:
                raise RuntimeError("No vectors stored in local mode")

            mapping = self._vector_mapping.get(idx)
            if mapping and "local_id" in mapping:
                local_id = mapping["local_id"]
                return self._vectors[local_id]
            else:
                raise IndexError(f"Vector mapping not found for index {idx}")

    def reconstruct_n(self, indices: List[int]) -> np.ndarray:
        """
        Reconstruct multiple binary vectors by their indices.

        Args:
            indices (List[int]): Indices of vectors to reconstruct

        Returns:
            np.ndarray: The reconstructed binary vectors with shape (len(indices), d)
        """
        vectors = np.zeros((len(indices), self.d), dtype=np.uint8)
        for i, idx in enumerate(indices):
            vectors[i] = self.reconstruct(idx)
        return vectors

    def reset(self) -> None:
        """
        Reset the index, removing all vectors but keeping the training.
        """
        self.ntotal = 0
        self._vectors = None
        self._vector_mapping = {}

        # Reset hash tables but keep bit masks
        self.hash_tables = [{} for _ in range(self.nhash)]

        # Get client for potential remote operations
        client = get_client()

        if client and client.is_remote_mode():
            # Remote mode: reset on server
            # Placeholder for server implementation - to be completed
            client.reset_index(self.index_id)
