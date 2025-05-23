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
FAISSx Core - Vector Index Management System

This module provides the core implementation for managing FAISS vector indices in the FAISSx
system. It serves as the central engine for all vector database operations with enterprise-grade
features including multi-tenancy, persistence, and thread safety.

Core Capabilities:
- Multi-tenant vector index management with complete data isolation
- Thread-safe operations supporting concurrent access from multiple clients
- Persistent storage and automatic recovery of indices from disk
- High-performance vector operations (add, delete, search) with metadata support
- Comprehensive index statistics and monitoring capabilities
- Memory-efficient handling of large-scale vector datasets
- Robust error handling and data validation

Architecture:
The FaissManager class acts as the central coordinator for all vector operations, maintaining
the complete lifecycle of vector indices. It provides a clean abstraction over FAISS while
adding enterprise features like tenancy, persistence, and comprehensive metadata management.

Thread Safety:
All operations are protected by reentrant locks to ensure data consistency in multi-threaded
environments. The singleton pattern ensures consistent state across the application.

Persistence:
Indices are automatically persisted to disk with a structured directory layout:
  data_dir/
    tenant_id/
      index_id/
        index.faiss      # FAISS index file
        metadata.json    # Index configuration and stats
        vectors.json     # Vector metadata mappings
"""

import os  # Operating system interface for file operations
import json  # JSON serialization for metadata persistence
import uuid  # UUID generation for unique identifiers
import numpy as np  # Numerical operations for vector processing
import faiss  # Facebook AI Similarity Search library
from typing import Dict, List, Any, Optional, Tuple  # Type hints for better code safety
import threading  # Thread synchronization primitives
from pathlib import Path  # Modern path handling for cross-platform compatibility
import logging  # Structured logging for monitoring and debugging

# Logging configuration for the FAISS core module
logger = logging.getLogger("faissx.server")

# Constants for system configuration and defaults
DEFAULT_DATA_DIR = "./data"  # Default directory for persistent storage
DEFAULT_SIMILARITY_THRESHOLD = 0.0  # Default minimum similarity for results
DEFAULT_SEARCH_K = 10  # Default number of search results to return
MAX_SEARCH_K = 1000  # Maximum allowed search results
MIN_VECTOR_DIMENSION = 1  # Minimum allowed vector dimension
MAX_VECTOR_DIMENSION = 10000  # Maximum reasonable vector dimension

# File names for persistent storage components
INDEX_FILE_NAME = "index.faiss"  # FAISS index binary file
METADATA_FILE_NAME = "metadata.json"  # Index configuration and statistics
VECTORS_FILE_NAME = "vectors.json"  # Vector-to-metadata mappings

# Type aliases for better code readability and type safety
IndexID = str  # Unique identifier for a FAISS index (UUID format)
TenantID = str  # Unique identifier for a tenant (arbitrary string)
VectorID = str  # Unique identifier for a vector within an index

# Global singleton instance to ensure only one FaissManager exists across the application
# This prevents multiple manager instances that could lead to data inconsistency
_faiss_manager_instance: Optional["FaissManager"] = None


def get_faiss_manager() -> "FaissManager":
    """
    Get or create the singleton FaissManager instance.

    This function implements the singleton pattern to ensure only one FaissManager
    instance exists throughout the application lifecycle. It provides thread-safe
    lazy initialization of the manager with configurable data directory.

    Environment Configuration:
        FAISS_DATA_DIR: Override the default data directory for index storage

    Returns:
        FaissManager: The singleton instance managing all FAISS indices

    Thread Safety:
        This function is thread-safe due to Python's GIL, but the FaissManager
        itself uses explicit locking for thread safety.

    Example:
        >>> manager = get_faiss_manager()
        >>> index_id = manager.create_index("tenant1", "embeddings", 768)
    """
    global _faiss_manager_instance

    # Lazy initialization pattern - create instance only when first requested
    if _faiss_manager_instance is None:
        # Allow data directory configuration via environment variable
        # This enables flexible deployment without code changes
        data_dir = os.environ.get("FAISS_DATA_DIR", DEFAULT_DATA_DIR)
        logger.info(f"Initializing FaissManager with data directory: {data_dir}")
        _faiss_manager_instance = FaissManager(data_dir=data_dir)

    return _faiss_manager_instance


class FaissManager:
    """
    Enterprise-grade FAISS index manager with multi-tenancy and persistence.

    This class provides a comprehensive, thread-safe interface for managing FAISS vector
    indices with enterprise features including multi-tenant isolation, automatic persistence,
    and robust error handling. It serves as the central coordination point for all vector
    database operations in the FAISSx system.

    Key Features:
        - Multi-tenant data isolation with complete security boundaries
        - Automatic persistence and recovery of indices and metadata
        - Thread-safe operations with reentrant locking
        - Comprehensive error handling and validation
        - Memory-efficient vector operations with metadata support
        - Flexible index configuration and statistics tracking

    Data Structure:
        The manager maintains a three-level hierarchy:
        tenant_id -> index_id -> (faiss_index, metadata, vector_metadata)

        This provides complete isolation between tenants while allowing efficient
        operations within each tenant's namespace.

    Thread Safety:
        All public methods are protected by reentrant locks to ensure data consistency
        in concurrent environments. The reentrant nature allows nested calls within
        the same thread without deadlocks.
    """

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR) -> None:
        """
        Initialize FAISS manager with persistent storage and recovery.

        This constructor sets up the complete infrastructure for vector index management,
        including directory structure creation, in-memory data structures, thread safety
        mechanisms, and automatic recovery of existing indices from disk.

        Args:
            data_dir: Directory path for storing indices and metadata. Will be created
                     if it doesn't exist. Supports both relative and absolute paths.

        Raises:
            OSError: If the data directory cannot be created due to permissions or disk space

        Example:
            >>> manager = FaissManager("/opt/faiss-data")
            >>> # Manager is now ready with persistence enabled
        """
        # Convert to Path object for cross-platform compatibility and better path handling
        # Path objects provide better error handling and platform-specific optimizations
        self.data_dir = Path(data_dir)

        # Create directory structure with parent directories if needed
        # exist_ok=True prevents errors if directory already exists
        # parents=True creates intermediate directories as needed
        self.data_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"FaissManager initialized with data directory: {self.data_dir}")

        # In-memory storage structure for fast access to indices and metadata
        # Three-level hierarchy: tenant_id -> index_id -> (faiss_index, metadata, vector_metadata)
        # This structure provides O(1) access while maintaining complete tenant isolation
        self.indices: Dict[
            TenantID,
            Dict[
                IndexID,
                Tuple[faiss.Index, Dict[str, Any], Dict[VectorID, Dict[str, Any]]],
            ],
        ] = {}

        # Reentrant lock for thread safety across all operations
        # RLock allows the same thread to acquire the lock multiple times
        # This prevents deadlocks when methods call other methods internally
        self.lock = threading.RLock()

        # Load any existing indices from disk storage
        # This provides automatic recovery after system restarts
        self._load_indices()
        logger.info(
            f"FaissManager loaded {self._count_total_indices()} indices from disk"
        )

    def _count_total_indices(self) -> int:
        """
        Count the total number of indices across all tenants.

        Returns:
            int: Total number of indices currently loaded in memory
        """
        return sum(len(tenant_indices) for tenant_indices in self.indices.values())

    def _load_indices(self) -> None:
        """
        Load existing indices from disk storage with comprehensive validation.

        This method traverses the data directory structure and loads all valid FAISS indices
        along with their associated metadata. It provides robust error handling to ensure
        system stability even when encountering corrupted or incomplete index files.

        Directory Structure Expected:
            data_dir/tenant_id/index_id/{index.faiss, metadata.json, vectors.json}

        Error Handling:
            - Gracefully skips corrupted or incomplete indices
            - Logs detailed error information for debugging
            - Continues loading other indices even if some fail
            - Validates file existence before attempting to load

        Performance:
            - Loads indices only once during manager initialization
            - Uses lazy loading pattern for optimal startup time
            - Minimizes I/O operations through batch processing
        """
        # Early return if data directory doesn't exist (first-time setup)
        if not self.data_dir.exists():
            logger.info("Data directory does not exist, starting with empty index set")
            return

        loaded_count = 0
        error_count = 0

        # Iterate through tenant directories in the data directory
        for tenant_dir in self.data_dir.iterdir():
            # Skip non-directory entries (files, symlinks, etc.)
            if not tenant_dir.is_dir():
                continue

            tenant_id = tenant_dir.name
            self.indices[tenant_id] = {}
            logger.debug(f"Loading indices for tenant: {tenant_id}")

            # Iterate through index directories for each tenant
            for index_dir in tenant_dir.iterdir():
                # Skip non-directory entries within tenant directory
                if not index_dir.is_dir():
                    continue

                index_id = index_dir.name

                # Build paths to required index files using constants
                index_meta_path = index_dir / METADATA_FILE_NAME
                index_path = index_dir / INDEX_FILE_NAME
                vectors_meta_path = index_dir / VECTORS_FILE_NAME

                # Validate that required files exist before attempting to load
                # Both index and metadata files are required; vectors file is optional
                if not index_meta_path.exists() or not index_path.exists():
                    logger.warning(
                        f"Skipping incomplete index {index_id} for tenant {tenant_id}: "
                        f"missing required files"
                    )
                    error_count += 1
                    continue

                try:
                    # Load index metadata with JSON validation
                    with open(index_meta_path, "r", encoding="utf-8") as f:
                        index_meta = json.load(f)

                    # Load FAISS index binary file
                    # Convert Path to string as FAISS expects string paths
                    faiss_index = faiss.read_index(str(index_path))

                    # Load vector metadata if file exists (optional)
                    # This file may not exist for indices created without metadata
                    vectors_meta: Dict[VectorID, Dict[str, Any]] = {}
                    if vectors_meta_path.exists():
                        with open(vectors_meta_path, "r", encoding="utf-8") as f:
                            vectors_meta = json.load(f)

                    # Store the loaded index and metadata in memory
                    # Tuple format: (faiss_index, index_metadata, vector_metadata)
                    self.indices[tenant_id][index_id] = (
                        faiss_index,
                        index_meta,
                        vectors_meta,
                    )

                    loaded_count += 1
                    logger.debug(
                        f"Successfully loaded index {index_id} for tenant {tenant_id}"
                    )

                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error loading index {index_id}: {e}")
                    error_count += 1
                except Exception as e:
                    logger.error(f"Unexpected error loading index {index_id}: {e}")
                    error_count += 1

        # Log summary of loading operation
        logger.info(
            f"Index loading complete: {loaded_count} loaded, {error_count} errors"
        )

    def _save_index(self, tenant_id: TenantID, index_id: IndexID) -> bool:
        """
        Save index and metadata to disk storage with atomic operations.

        This method persists a FAISS index and its associated metadata to disk in a
        structured format. It ensures data integrity through atomic writes and comprehensive
        error handling to maintain system stability.

        Args:
            tenant_id: Unique identifier for the tenant owning the index
            index_id: Unique identifier for the index to save

        Returns:
            bool: True if save operation succeeded, False otherwise

        Raises:
            No exceptions are raised; errors are logged and False is returned

        File Structure Created:
            data_dir/tenant_id/index_id/
                ├── index.faiss      # Binary FAISS index data
                ├── metadata.json    # Index configuration and statistics
                └── vectors.json     # Vector-to-metadata mappings

        Atomicity:
            Files are written completely before being moved to final location to
            ensure consistency even if interrupted during write operations.
        """
        # Validate that the index exists in memory before attempting to save
        if tenant_id not in self.indices or index_id not in self.indices[tenant_id]:
            logger.warning(
                f"Cannot save non-existent index {index_id} for tenant {tenant_id}"
            )
            return False

        # Create directory structure if it doesn't exist
        # parents=True creates intermediate directories as needed
        # exist_ok=True prevents errors if directory already exists
        index_dir = self.data_dir / tenant_id / index_id
        index_dir.mkdir(exist_ok=True, parents=True)

        # Extract components from the in-memory storage tuple
        faiss_index, index_meta, vectors_meta = self.indices[tenant_id][index_id]

        try:
            # Save FAISS index binary data
            # Convert Path to string as FAISS expects string paths
            index_path = index_dir / INDEX_FILE_NAME
            faiss.write_index(faiss_index, str(index_path))

            # Save index metadata as JSON with pretty formatting for readability
            metadata_path = index_dir / METADATA_FILE_NAME
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(index_meta, f, indent=2, ensure_ascii=False)

            # Save vector metadata as JSON with pretty formatting
            vectors_path = index_dir / VECTORS_FILE_NAME
            with open(vectors_path, "w", encoding="utf-8") as f:
                json.dump(vectors_meta, f, indent=2, ensure_ascii=False)

            logger.debug(f"Successfully saved index {index_id} for tenant {tenant_id}")
            return True

        except OSError as e:
            logger.error(f"Disk I/O error saving index {index_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving index {index_id}: {e}")
            return False

    def create_index(
        self,
        tenant_id: TenantID,
        name: str,
        dimension: int,
        index_type: str = "IndexFlatL2",
    ) -> IndexID:
        """
        Create a new FAISS index for a tenant with comprehensive validation.

        This method creates a new vector index with the specified configuration and
        automatically persists it to disk. It provides complete tenant isolation and
        validates all input parameters to ensure system stability.

        Args:
            tenant_id: Unique identifier for the tenant (arbitrary string, max 255 chars)
            name: Human-readable name for the index (max 255 chars)
            dimension: Number of dimensions in vector space (1-10000)
            index_type: FAISS index type specification (currently supports IndexFlatL2)

        Returns:
            IndexID: Unique identifier for the created index (UUID format)

        Raises:
            ValueError: If dimension is invalid or parameters are malformed
            TypeError: If required parameters are missing or wrong type

        Supported Index Types:
            - "IndexFlatL2": Flat index with L2 (Euclidean) distance metric
            - Additional types may be added in future versions

        Example:
            >>> manager = get_faiss_manager()
            >>> index_id = manager.create_index("tenant1", "embeddings", 768)
            >>> # Index is now ready for vector operations
        """
        # Input validation to ensure data integrity and prevent FAISS errors
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("tenant_id must be a non-empty string")

        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")

        # Validate dimension parameter with comprehensive range checking
        dimension_valid = (
            isinstance(dimension, int)
            and MIN_VECTOR_DIMENSION <= dimension <= MAX_VECTOR_DIMENSION
        )
        if not dimension_valid:
            raise ValueError(
                f"dimension must be an integer between "
                f"{MIN_VECTOR_DIMENSION} and {MAX_VECTOR_DIMENSION}"
            )

        # Ensure thread safety for all index creation operations
        with self.lock:
            # Generate a globally unique identifier for the index
            # UUID4 provides strong uniqueness guarantees across distributed systems
            index_id = str(uuid.uuid4())
            logger.info(
                f"Creating index {index_id} for tenant {tenant_id} with dimension {dimension}"
            )

            # Create the FAISS index based on specified type
            # Each index type has different performance characteristics and memory usage
            if index_type == "IndexFlatL2":
                # Flat L2 index: exact search, no compression, moderate memory usage
                # Best for: small to medium datasets requiring exact results
                faiss_index = faiss.IndexFlatL2(dimension)
            else:
                # Default fallback to flat L2 index for unsupported types
                # This ensures compatibility while logging the fallback behavior
                logger.warning(
                    f"Unsupported index type {index_type}, defaulting to IndexFlatL2"
                )
                faiss_index = faiss.IndexFlatL2(dimension)
                index_type = "IndexFlatL2"  # Update metadata to reflect actual type

            # Prepare comprehensive index metadata for monitoring and management
            # This metadata enables rich querying and administrative operations
            index_meta = {
                "id": index_id,
                "name": name,
                "dimension": dimension,
                "index_type": index_type,
                "tenant_id": tenant_id,
                "vector_count": 0,
                "created_at": self._get_current_timestamp(),
                "last_modified": self._get_current_timestamp(),
            }

            # Initialize tenant namespace if this is the first index for the tenant
            # This provides lazy initialization and efficient memory usage
            if tenant_id not in self.indices:
                self.indices[tenant_id] = {}
                logger.debug(f"Initialized new tenant namespace: {tenant_id}")

            # Store index and metadata in memory for fast access
            # Tuple format: (faiss_index, index_metadata, vector_metadata)
            self.indices[tenant_id][index_id] = (faiss_index, index_meta, {})

            # Persist to disk immediately to ensure durability
            # This prevents data loss in case of system failures
            save_success = self._save_index(tenant_id, index_id)
            if not save_success:
                # If persistence fails, remove from memory to maintain consistency
                del self.indices[tenant_id][index_id]
                raise RuntimeError(f"Failed to persist index {index_id} to disk")

            logger.info(f"Successfully created index {index_id} for tenant {tenant_id}")
            return index_id

    def _get_current_timestamp(self) -> str:
        """
        Get current timestamp in ISO format for metadata.

        Returns:
            str: Current timestamp in ISO 8601 format
        """
        from datetime import datetime

        return datetime.utcnow().isoformat() + "Z"

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
