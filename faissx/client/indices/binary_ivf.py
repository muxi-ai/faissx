#!/usr/bin/env python3
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
Binary IVF Index implementation for FAISSx.

This module provides a binary vector index using Inverted File (IVF) with Hamming distance,
which is compatible with the FAISS IndexBinaryIVF.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from faissx.client.client import get_client
from faissx.client.indices.binary_base import BinaryIndex
from faissx.client.indices.binary_flat import IndexBinaryFlat


class IndexBinaryIVF(BinaryIndex):
    """
    IndexBinaryIVF is a binary vector index that uses Inverted File (IVF) with Hamming distance.

    This index uses coarse quantization to speed up search by partitioning the vector space
    and only searching a subset of partitions during queries. It is compatible with the FAISS
    IndexBinaryIVF implementation.
    """

    def __init__(self, index: IndexBinaryFlat, nlist: int):
        """
        Initialize the binary IVF index.

        Args:
            index (IndexBinaryFlat): The quantizer to use for coarse quantization.
                                     Must be an IndexBinaryFlat.
            nlist (int): Number of clusters/lists in the inverted file.
        """
        super().__init__(index.d)  # Initialize binary base class with same dimension

        # Store parameters
        self.nlist: int = nlist
        self.quantizer = index
        self.is_trained: bool = False  # IVF indices need training

        # Search-time parameter
        self.nprobe: int = 1  # Number of clusters to search at query time (default: 1)

        # Local storage for inverted lists
        self._invlists: List[List[int]] = []  # Lists of vector indices for each cluster
        self._centroids: Optional[np.ndarray] = None  # Cluster centroids
        self._assigned_clusters: Dict[int, int] = {}  # Map from vector ID to cluster ID

    def train(self, vectors: np.ndarray) -> None:
        """
        Train the index by clustering the input vectors.

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
            # Local mode: perform k-means clustering with Hamming distance
            # This is a simplified implementation for demonstration
            n = vectors.shape[0]

            if n < self.nlist:
                # Not enough vectors for training with the requested nlist
                raise ValueError(f"Cannot train with {n} vectors and nlist={self.nlist}. "
                                 f"Need at least {self.nlist} training vectors.")

            # Randomly select initial centroids
            centroid_indices = np.random.choice(n, size=self.nlist, replace=False)
            self._centroids = vectors[centroid_indices].copy()

            # Perform k-means iterations
            max_iterations = 10
            for _ in range(max_iterations):
                # Assign vectors to clusters
                distances = self.compute_hamming_distances_optimized(vectors, self._centroids)
                assignments = np.argmin(distances, axis=1)

                # Update centroids
                new_centroids = np.zeros_like(self._centroids)
                changed = False

                for i in range(self.nlist):
                    # Find vectors assigned to this cluster
                    cluster_vectors = vectors[assignments == i]

                    if len(cluster_vectors) > 0:
                        # Compute new centroid (this is approximate for binary vectors)
                        # For binary vectors, we use a majority vote for each bit
                        votes = np.mean(np.unpackbits(cluster_vectors.reshape(-1, 1)), axis=0)
                        majority = (votes > 0.5).astype(np.uint8)
                        new_centroid = np.packbits(majority).reshape(1, -1)[:, :self.d]

                        if not np.array_equal(new_centroid[0], self._centroids[i]):
                            new_centroids[i] = new_centroid[0]
                            changed = True
                        else:
                            new_centroids[i] = self._centroids[i]
                    else:
                        # Keep the old centroid if no vectors assigned
                        new_centroids[i] = self._centroids[i]

                self._centroids = new_centroids

                # Break if centroids didn't change
                if not changed:
                    break

            # Initialize empty inverted lists
            self._invlists = [[] for _ in range(self.nlist)]

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
            # Local mode: store vectors and assign to clusters
            if self._vectors is None:
                self._vectors = vectors.copy()
            else:
                self._vectors = np.vstack((self._vectors, vectors))

            # Assign vectors to clusters
            distances = self.compute_hamming_distances_optimized(vectors, self._centroids)
            assignments = np.argmin(distances, axis=1)

            # Add vectors to inverted lists
            for i in range(n):
                vector_idx = start_idx + i
                cluster_idx = assignments[i]

                # Store in inverted list
                self._invlists[cluster_idx].append(vector_idx)

                # Update cluster assignment map
                self._assigned_clusters[vector_idx] = cluster_idx

                # Update vector mapping for local mode
                self._vector_mapping[vector_idx] = {"local_id": vector_idx}

        # Update total count
        self.ntotal += n

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors using Hamming distance with IVF.

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
            # Local mode: search using inverted file
            if self.ntotal == 0 or self._vectors is None:
                # Return empty results if no vectors
                n = query.shape[0]
                return np.zeros((n, k), dtype=np.float32), np.zeros((n, k), dtype=np.int64)

            n_queries = query.shape[0]
            nprobe = min(self.nprobe, self.nlist)  # Ensure nprobe isn't larger than nlist

            # Initialize results
            results_distances = np.full((n_queries, k), np.inf, dtype=np.float32)
            results_indices = np.zeros((n_queries, k), dtype=np.int64)

            for i in range(n_queries):
                # Find closest clusters to query
                distances_to_centroids = self.compute_hamming_distances(
                    query[i:i+1], self._centroids
                )[0]

                # Get top nprobe clusters
                top_clusters = np.argsort(distances_to_centroids)[:nprobe]

                # Collect candidate vector indices from selected clusters
                candidates = []
                for cluster_idx in top_clusters:
                    candidates.extend(self._invlists[cluster_idx])

                if not candidates:
                    continue  # No candidates found

                # Compute distances to all candidates
                candidate_vectors = self._vectors[candidates]
                candidate_distances = self.compute_hamming_distances(
                    query[i:i+1], candidate_vectors
                )[0]

                # Get top k results
                if len(candidates) <= k:
                    # If fewer candidates than k, return all
                    top_k = np.argsort(candidate_distances)
                    results_distances[i, :len(candidates)] = candidate_distances[top_k]
                    results_indices[i, :len(candidates)] = [candidates[j] for j in top_k]
                else:
                    # Get top k
                    top_k = np.argpartition(candidate_distances, k)[:k]
                    # Sort the top k
                    sorted_top_k = top_k[np.argsort(candidate_distances[top_k])]
                    results_distances[i, :k] = candidate_distances[sorted_top_k]
                    results_indices[i, :k] = [candidates[j] for j in sorted_top_k]

            return results_distances, results_indices

    def set_nprobe(self, nprobe: int) -> None:
        """
        Set the number of clusters to visit during search.

        Args:
            nprobe (int): Number of clusters to search
        """
        if nprobe <= 0:
            raise ValueError(f"nprobe must be positive, got {nprobe}")
        self.nprobe = nprobe

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
        self._invlists = [[] for _ in range(self.nlist)]
        self._assigned_clusters = {}

        # Get client for potential remote operations
        client = get_client()

        if client and client.is_remote_mode():
            # Remote mode: reset on server
            # Placeholder for server implementation - to be completed
            client.reset_index(self.index_id)
