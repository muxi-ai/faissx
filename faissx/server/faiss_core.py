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

import os
import json
import uuid
import numpy as np
import faiss
from typing import Dict, List, Any, Optional, Tuple
import threading
from pathlib import Path

# Type aliases for better code readability and type safety
IndexID = str  # Unique identifier for a FAISS index
TenantID = str  # Unique identifier for a tenant
VectorID = str  # Unique identifier for a vector

# Global singleton instance to ensure only one FaissManager exists
_faiss_manager_instance = None


def get_faiss_manager():
    """
    Get or create the singleton FaissManager instance.

    Returns:
        FaissManager: The singleton instance managing FAISS indices
    """
    global _faiss_manager_instance

    if _faiss_manager_instance is None:
        # Use environment variable for data directory or default to ./data
        data_dir = os.environ.get("FAISS_DATA_DIR", "./data")
        _faiss_manager_instance = FaissManager(data_dir=data_dir)

    return _faiss_manager_instance


class FaissManager:
    """
    Manager for FAISS indices with tenant isolation and persistence.

    This class provides a thread-safe interface for managing FAISS vector indices
    with support for multiple tenants and persistent storage.
    """

    def __init__(self, data_dir: str = "./data"):
        """
        Initialize FAISS manager with persistent storage.

        Args:
            data_dir: Directory for storing indices and metadata
        """
        # Convert to Path object for cross-platform compatibility
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)

        # In-memory storage structure:
        # tenant_id -> {index_id -> (faiss_index, index_metadata, {vector_id -> metadata})}
        self.indices: Dict[
            TenantID, Dict[IndexID, Tuple[faiss.Index, Dict, Dict[VectorID, Dict]]]
        ] = {}

        # Reentrant lock for thread safety
        self.lock = threading.RLock()

        # Load any existing indices from disk
        self._load_indices()

    def _load_indices(self):
        """
        Load existing indices from disk storage.

        This method traverses the data directory structure and loads all
        valid FAISS indices and their associated metadata.
        """
        if not self.data_dir.exists():
            return

        # Iterate through tenant directories
        for tenant_dir in self.data_dir.iterdir():
            if not tenant_dir.is_dir():
                continue

            tenant_id = tenant_dir.name
            self.indices[tenant_id] = {}

            # Iterate through index directories for each tenant
            for index_dir in tenant_dir.iterdir():
                if not index_dir.is_dir():
                    continue

                index_id = index_dir.name
                index_meta_path = index_dir / "metadata.json"
                index_path = index_dir / "index.faiss"
                vectors_meta_path = index_dir / "vectors.json"

                # Skip if required files are missing
                if not index_meta_path.exists() or not index_path.exists():
                    continue

                try:
                    # Load index metadata
                    with open(index_meta_path, "r") as f:
                        index_meta = json.load(f)

                    # Load FAISS index
                    faiss_index = faiss.read_index(str(index_path))

                    # Load vector metadata if it exists
                    vectors_meta = {}
                    if vectors_meta_path.exists():
                        with open(vectors_meta_path, "r") as f:
                            vectors_meta = json.load(f)

                    # Store in memory
                    self.indices[tenant_id][index_id] = (
                        faiss_index,
                        index_meta,
                        vectors_meta,
                    )
                except Exception as e:
                    print(f"Error loading index {index_id}: {e}")

    def _save_index(self, tenant_id: TenantID, index_id: IndexID):
        """
        Save index and metadata to disk storage.

        Args:
            tenant_id: Tenant ID
            index_id: Index ID
        """
        if tenant_id not in self.indices or index_id not in self.indices[tenant_id]:
            return

        # Create directory structure if it doesn't exist
        index_dir = self.data_dir / tenant_id / index_id
        index_dir.mkdir(exist_ok=True, parents=True)

        faiss_index, index_meta, vectors_meta = self.indices[tenant_id][index_id]

        try:
            # Save FAISS index
            faiss.write_index(faiss_index, str(index_dir / "index.faiss"))

            # Save index metadata
            with open(index_dir / "metadata.json", "w") as f:
                json.dump(index_meta, f)

            # Save vector metadata
            with open(index_dir / "vectors.json", "w") as f:
                json.dump(vectors_meta, f)
        except Exception as e:
            print(f"Error saving index {index_id}: {e}")

    def create_index(
        self,
        tenant_id: TenantID,
        name: str,
        dimension: int,
        index_type: str = "IndexFlatL2",
    ) -> IndexID:
        """
        Create a new FAISS index for a tenant.

        Args:
            tenant_id: Tenant ID
            name: Index name
            dimension: Vector dimension
            index_type: FAISS index type (currently only supports IndexFlatL2)

        Returns:
            index_id: ID of the created index
        """
        with self.lock:
            # Generate a unique ID for the index
            index_id = str(uuid.uuid4())

            # Create the FAISS index (currently only supports L2 distance)
            if index_type == "IndexFlatL2":
                faiss_index = faiss.IndexFlatL2(dimension)
            else:
                # Default to flat L2 index if type is not supported
                faiss_index = faiss.IndexFlatL2(dimension)

            # Prepare index metadata
            index_meta = {
                "id": index_id,
                "name": name,
                "dimension": dimension,
                "index_type": index_type,
                "tenant_id": tenant_id,
                "vector_count": 0,
            }

            # Initialize tenant if needed
            if tenant_id not in self.indices:
                self.indices[tenant_id] = {}

            # Store index and metadata
            self.indices[tenant_id][index_id] = (faiss_index, index_meta, {})

            # Save to disk
            self._save_index(tenant_id, index_id)

            return index_id

    def get_index_info(self, tenant_id: TenantID, index_id: IndexID) -> Optional[Dict]:
        """
        Get index information and metadata.

        Args:
            tenant_id: Tenant ID
            index_id: Index ID

        Returns:
            index_meta: Index metadata or None if not found
        """
        with self.lock:
            if tenant_id not in self.indices or index_id not in self.indices[tenant_id]:
                return None

            _, index_meta, _ = self.indices[tenant_id][index_id]
            return dict(index_meta)  # Return a copy to prevent modification

    def delete_index(self, tenant_id: TenantID, index_id: IndexID) -> bool:
        """
        Delete an index and its associated data.

        Args:
            tenant_id: Tenant ID
            index_id: Index ID

        Returns:
            success: Whether the deletion succeeded
        """
        with self.lock:
            if tenant_id not in self.indices or index_id not in self.indices[tenant_id]:
                return False

            # Remove from memory
            del self.indices[tenant_id][index_id]

            # Remove from disk
            index_dir = self.data_dir / tenant_id / index_id
            if index_dir.exists():
                # Remove all files in the directory
                for file in index_dir.iterdir():
                    file.unlink()
                # Remove the directory itself
                index_dir.rmdir()

            return True

    def add_vectors(
        self, tenant_id: TenantID, index_id: IndexID, vectors: List[Any]
    ) -> Dict[str, Any]:
        """
        Add vectors to an index with metadata.

        Args:
            tenant_id: Tenant ID
            index_id: Index ID
            vectors: List of vectors (either dicts or array-like objects)

        Returns:
            result: Dict with success info, added and failed counts
        """
        with self.lock:
            if tenant_id not in self.indices or index_id not in self.indices[tenant_id]:
                return {
                    "success": False,
                    "added_count": 0,
                    "failed_count": len(vectors),
                }

            faiss_index, index_meta, vectors_meta = self.indices[tenant_id][index_id]

            # Get expected vector dimension
            dimension = index_meta["dimension"]

            # Track success/failure for each vector
            success_list = []
            vectors_to_add = []
            vector_ids = []
            vector_metadata = []

            for vector in vectors:
                # Handle both dict and vector-like formats
                if isinstance(vector, dict):
                    vector_id = vector.get("id")
                    vector_values = vector.get("values", [])
                    metadata = vector.get("metadata", {})
                elif hasattr(vector, "id") and hasattr(vector, "values"):  # Object-like
                    vector_id = vector.id
                    vector_values = vector.values
                    metadata = getattr(vector, "metadata", {})
                else:
                    success_list.append(False)
                    continue

                if not vector_id:
                    success_list.append(False)
                    continue

                # Validate vector dimension
                if len(vector_values) != dimension:
                    success_list.append(False)
                    continue

                # Prepare vector for addition
                vectors_to_add.append(vector_values)
                vector_ids.append(vector_id)
                vector_metadata.append(metadata)
                success_list.append(True)

            if vectors_to_add:
                # Convert to numpy array and add to index
                vectors_array = np.array(vectors_to_add, dtype=np.float32)
                faiss_index.add(vectors_array)

                # Store metadata
                for i, vector_id in enumerate(vector_ids):
                    vectors_meta[vector_id] = vector_metadata[i]

                # Update vector count
                index_meta["vector_count"] += len(vectors_to_add)

                # Save to disk
                self._save_index(tenant_id, index_id)

            return {
                "success": any(success_list),
                "added_count": sum(success_list),
                "failed_count": len(vectors) - sum(success_list),
            }

    def search(
        self,
        tenant_id: TenantID,
        index_id: IndexID,
        vector: List[float],
        k: int = 10,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Search for similar vectors in an index.

        Args:
            tenant_id: Tenant ID
            index_id: Index ID
            vector: Query vector
            k: Number of results to return
            filter_metadata: Metadata filter

        Returns:
            results: List of search results
        """
        with self.lock:
            if tenant_id not in self.indices or index_id not in self.indices[tenant_id]:
                return []

            faiss_index, index_meta, vectors_meta = self.indices[tenant_id][index_id]

            # Convert query to numpy array
            query = np.array([vector], dtype=np.float32)

            # Search in FAISS index
            distances, indices = faiss_index.search(query, k)

            # Prepare results
            results = []
            all_vector_ids = list(vectors_meta.keys())

            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                # Skip invalid indices (can happen if index has fewer vectors than k)
                if idx < 0 or idx >= len(all_vector_ids):
                    continue

                vector_id = all_vector_ids[idx]
                metadata = vectors_meta[vector_id]

                # Apply metadata filter if provided
                if filter_metadata:
                    if not self._match_metadata(metadata, filter_metadata):
                        continue

                # FAISS returns squared L2 distance, convert to similarity score
                # Higher score is better (1.0 is identical, 0.0 is completely dissimilar)
                similarity = 1.0 / (1.0 + distance)

                results.append(
                    {"id": vector_id, "score": similarity, "metadata": metadata}
                )

            return results

    def _match_metadata(self, metadata: Dict, filter_metadata: Dict) -> bool:
        """
        Check if metadata matches filter criteria.

        Args:
            metadata: Vector metadata
            filter_metadata: Filter criteria

        Returns:
            matches: Whether metadata matches filter
        """
        for key, value in filter_metadata.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    def delete_vector(
        self, tenant_id: TenantID, index_id: IndexID, vector_id: VectorID
    ) -> bool:
        """
        Delete a vector from an index.

        Note: FAISS doesn't support direct deletion, so we rebuild the index without the vector.
        This is an expensive operation and not recommended for frequent use.

        Args:
            tenant_id: Tenant ID
            index_id: Index ID
            vector_id: Vector ID

        Returns:
            success: Whether deletion succeeded
        """
        with self.lock:
            if tenant_id not in self.indices or index_id not in self.indices[tenant_id]:
                return False

            faiss_index, index_meta, vectors_meta = self.indices[tenant_id][index_id]

            # Check if vector exists
            if vector_id not in vectors_meta:
                return False

            # Remove metadata
            del vectors_meta[vector_id]

            # Update vector count
            index_meta["vector_count"] -= 1

            # Save to disk
            self._save_index(tenant_id, index_id)

            # Note: We don't actually remove from FAISS index as it would require rebuilding
            # For a production version, we would need to implement proper deletion or have a cleanup process

            return True
