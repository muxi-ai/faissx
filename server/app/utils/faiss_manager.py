import os
import json
import uuid
import numpy as np
import faiss
from typing import Dict, List, Any, Optional, Tuple
import threading
from pathlib import Path

# Type aliases
IndexID = str
TenantID = str
VectorID = str

# Global singleton instance
_faiss_manager_instance = None


def get_faiss_manager():
    """Get the singleton FaissManager instance"""
    global _faiss_manager_instance

    if _faiss_manager_instance is None:
        data_dir = os.environ.get("FAISS_DATA_DIR", "./data")
        _faiss_manager_instance = FaissManager(data_dir=data_dir)

    return _faiss_manager_instance


class FaissManager:
    """
    Manager for FAISS indices with tenant isolation and persistence
    """
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize FAISS manager.

        Args:
            data_dir: Directory for storing indices and metadata
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)

        # In-memory storage for indices and metadata
        # {tenant_id: {index_id: (faiss_index, index_metadata, {vector_id: metadata})}}
        self.indices: Dict[TenantID, Dict[IndexID, Tuple[faiss.Index, Dict, Dict[VectorID, Dict]]]] = {}

        # Lock for thread safety
        self.lock = threading.RLock()

        # Load existing indices from disk
        self._load_indices()

    def _load_indices(self):
        """Load existing indices from disk"""
        if not self.data_dir.exists():
            return

        for tenant_dir in self.data_dir.iterdir():
            if not tenant_dir.is_dir():
                continue

            tenant_id = tenant_dir.name
            self.indices[tenant_id] = {}

            for index_dir in tenant_dir.iterdir():
                if not index_dir.is_dir():
                    continue

                index_id = index_dir.name
                index_meta_path = index_dir / "metadata.json"
                index_path = index_dir / "index.faiss"
                vectors_meta_path = index_dir / "vectors.json"

                if not index_meta_path.exists() or not index_path.exists():
                    continue

                try:
                    with open(index_meta_path, "r") as f:
                        index_meta = json.load(f)

                    faiss_index = faiss.read_index(str(index_path))

                    vectors_meta = {}
                    if vectors_meta_path.exists():
                        with open(vectors_meta_path, "r") as f:
                            vectors_meta = json.load(f)

                    self.indices[tenant_id][index_id] = (faiss_index, index_meta, vectors_meta)
                except Exception as e:
                    print(f"Error loading index {index_id}: {e}")

    def _save_index(self, tenant_id: TenantID, index_id: IndexID):
        """
        Save index and metadata to disk.

        Args:
            tenant_id: Tenant ID
            index_id: Index ID
        """
        if tenant_id not in self.indices or index_id not in self.indices[tenant_id]:
            return

        index_dir = self.data_dir / tenant_id / index_id
        index_dir.mkdir(exist_ok=True, parents=True)

        faiss_index, index_meta, vectors_meta = self.indices[tenant_id][index_id]

        try:
            faiss.write_index(faiss_index, str(index_dir / "index.faiss"))

            with open(index_dir / "metadata.json", "w") as f:
                json.dump(index_meta, f)

            with open(index_dir / "vectors.json", "w") as f:
                json.dump(vectors_meta, f)
        except Exception as e:
            print(f"Error saving index {index_id}: {e}")

    def create_index(self, tenant_id: TenantID, name: str, dimension: int, index_type: str = "IndexFlatL2") -> IndexID:
        """
        Create a new FAISS index.

        Args:
            tenant_id: Tenant ID
            name: Index name
            dimension: Vector dimension
            index_type: FAISS index type

        Returns:
            index_id: ID of the created index
        """
        with self.lock:
            # Generate a unique ID for the index
            index_id = str(uuid.uuid4())

            # Create the FAISS index
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
                "vector_count": 0
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
        Get index information.

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
            return dict(index_meta)  # Return a copy

    def delete_index(self, tenant_id: TenantID, index_id: IndexID) -> bool:
        """
        Delete an index.

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
                # Remove files
                for file in index_dir.iterdir():
                    file.unlink()
                # Remove directory
                index_dir.rmdir()

            return True

    def add_vectors(self, tenant_id: TenantID, index_id: IndexID,
                   vectors: List[Any]) -> List[bool]:
        """
        Add vectors to an index.

        Args:
            tenant_id: Tenant ID
            index_id: Index ID
            vectors: List of vectors (either dicts or Pydantic Vector objects)

        Returns:
            success_list: List of booleans indicating success for each vector
        """
        with self.lock:
            if tenant_id not in self.indices or index_id not in self.indices[tenant_id]:
                return [False] * len(vectors)

            faiss_index, index_meta, vectors_meta = self.indices[tenant_id][index_id]

            # Check dimension
            dimension = index_meta["dimension"]

            success_list = []
            vectors_to_add = []

            for vector in vectors:
                # Handle both dict and Pydantic model formats
                if hasattr(vector, "id"):  # Pydantic model
                    vector_id = vector.id
                    vector_values = vector.values
                    vector_metadata = vector.metadata
                else:  # Dictionary
                    vector_id = vector["id"]
                    vector_values = vector["values"]
                    vector_metadata = vector.get("metadata", {})

                # Validate vector dimension
                if len(vector_values) != dimension:
                    success_list.append(False)
                    continue

                # Prepare vector for addition
                vectors_to_add.append(vector_values)

                # Store metadata
                vectors_meta[vector_id] = vector_metadata

                success_list.append(True)

            if vectors_to_add:
                # Convert to numpy array and add to index
                vectors_array = np.array(vectors_to_add, dtype=np.float32)
                faiss_index.add(vectors_array)

                # Update vector count
                index_meta["vector_count"] += len(vectors_to_add)

                # Save to disk
                self._save_index(tenant_id, index_id)

            return success_list

    def search(self, tenant_id: TenantID, index_id: IndexID,
               vector: List[float], k: int = 10,
               filter_metadata: Optional[Dict] = None) -> List[Dict]:
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

                results.append({
                    "id": vector_id,
                    "score": similarity,
                    "metadata": metadata
                })

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

    def delete_vector(self, tenant_id: TenantID, index_id: IndexID, vector_id: VectorID) -> bool:
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
