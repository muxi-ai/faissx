#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FAISSx Server Persistence Module
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
Enterprise Persistence Infrastructure for FAISSx Server

This module provides comprehensive persistence capabilities for FAISS indices, supporting
multiple serialization formats, metadata management, and robust data recovery mechanisms.
It enables seamless index storage, transfer, and reconstruction across different deployment
environments with enterprise-grade reliability and performance.

Key Features:
- Multi-format index persistence (FAISS native, pickle fallback)
- Comprehensive metadata extraction and storage with JSON serialization
- Vector data backup and reconstruction capabilities
- Binary serialization for network transfer and remote deployment
- Trained parameter extraction for model reuse and optimization
- Automatic error recovery with graceful fallback mechanisms
- Support for all FAISS index types including IVF, HNSW, PQ, and composite indices

Persistence Workflow:
1. Index Analysis: Extract comprehensive metadata and parameters
2. Multi-format Storage: Save index data with multiple fallback options
3. Vector Backup: Optional vector reconstruction data for complete recovery
4. Metadata Archival: Structured JSON metadata for inspection and validation
5. Transfer Packaging: Binary serialization for network deployment

Index Type Support:
- Flat indices (IndexFlat, IndexBinaryFlat) with full reconstruction
- IVF indices (IndexIVF*) with quantizer parameter extraction
- HNSW indices with graph structure parameters and connectivity settings
- PQ indices (IndexPQ) with codebook and quantization parameters
- Composite indices (IndexPreTransform) with transformation chain preservation
- Binary indices with specialized bit-level storage optimization

Integration:
This module integrates with the FAISSx server infrastructure to provide reliable
data persistence across server restarts, tenant migrations, and system upgrades.
"""

import os
import time
import json
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, Union, List
import logging

# Third-party imports
import faiss
import numpy as np

# Logging configuration for persistence operations
logger = logging.getLogger("faissx.server")

# File extension constants for different persistence formats
# These constants ensure consistent file naming across the persistence system
DEFAULT_METADATA_EXTENSION = ".meta.json"  # JSON metadata files
DEFAULT_VECTORS_EXTENSION = ".vectors.npy"  # NumPy vector data files
DEFAULT_INDEX_EXTENSION = ".index"  # FAISS index binary files
DEFAULT_PICKLE_EXTENSION = ".pickle"  # Pickle fallback files
DEFAULT_PARAMS_EXTENSION = ".params.json"  # Trained parameter files

# Serialization configuration constants
MAX_VECTORS_DEFAULT = 1000  # Default maximum vectors for serialization
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100MB default max file size
PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL  # Use highest available pickle protocol
JSON_INDENT = 2  # Standard JSON indentation for readability

# Index type constants for validation and processing
SUPPORTED_INDEX_TYPES = {
    "IndexFlat",
    "IndexBinaryFlat",
    "IndexIVF",
    "IndexIVFFlat",
    "IndexIVFPQ",
    "IndexHNSW",
    "IndexHNSWFlat",
    "IndexPQ",
    "IndexPreTransform",
    "IndexIDMap",
}

# Error handling constants
PERSISTENCE_ERRORS = {
    "FILE_NOT_FOUND": "Index file not found",
    "SERIALIZATION_FAILED": "Failed to serialize index",
    "DESERIALIZATION_FAILED": "Failed to deserialize index",
    "INVALID_INDEX_TYPE": "Unsupported index type",
    "PARAMETER_EXTRACTION_FAILED": "Failed to extract parameters",
}


def _validate_index(index: Any) -> None:
    """
    Validate that the provided index is a supported FAISS index type.

    Args:
        index: The index to validate

    Raises:
        TypeError: If the index is not a valid FAISS index
        ValueError: If the index type is not supported
    """
    if not hasattr(index, "ntotal"):
        raise TypeError("Object does not appear to be a FAISS index")

    index_type = type(index).__name__
    if index_type not in SUPPORTED_INDEX_TYPES:
        logger.warning(f"Index type {index_type} may not be fully supported")


def _create_directory_if_needed(filepath: Path) -> None:
    """
    Create the directory for the filepath if it doesn't exist.

    Args:
        filepath: Path where the file will be saved
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {filepath.parent}")


def _extract_index_metadata(
    index: Any, metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from a FAISS index.

    Args:
        index: The FAISS index to analyze
        metadata: Additional metadata to merge

    Returns:
        dict: Comprehensive index metadata
    """
    # Base index information
    index_info = {
        "index_type": type(index).__name__,
        "dimension": getattr(index, "d", None),
        "ntotal": index.ntotal,
        "is_trained": getattr(index, "is_trained", True),
        "metric_type": getattr(index, "metric_type", None),
        "timestamp": time.time(),
    }

    # Add specialized index parameters based on type
    if isinstance(index, faiss.IndexIVF):
        index_info.update(
            {
                "nlist": index.nlist,
                "nprobe": index.nprobe,
                "quantizer_type": type(index.quantizer).__name__,
                "inverted_lists_type": (
                    type(index.invlists).__name__
                    if hasattr(index, "invlists")
                    else None
                ),
            }
        )
    elif isinstance(index, faiss.IndexHNSW):
        index_info.update(
            {
                "hnsw": {
                    "M": index.hnsw.M,
                    "ef_search": index.hnsw.efSearch,
                    "ef_construction": index.hnsw.efConstruction,
                    "max_level": getattr(index.hnsw, "max_level", None),
                    "level_generator_seed": getattr(
                        index.hnsw, "level_generator_seed", None
                    ),
                }
            }
        )
    elif isinstance(index, faiss.IndexPQ):
        index_info.update(
            {
                "pq": {
                    "M": index.pq.M,
                    "nbits": index.pq.nbits,
                    "is_trained": getattr(index.pq, "is_trained", False),
                }
            }
        )
    elif isinstance(index, faiss.IndexBinaryFlat):
        index_info.update({"is_binary": True, "dimension_bits": index.d})
    elif isinstance(index, faiss.IndexPreTransform):
        index_info.update(
            {
                "transforms": _extract_transform_chain_info(index),
                "base_index_type": type(index.index).__name__,
            }
        )

    # Merge with provided metadata
    if metadata:
        index_info.update(metadata)

    return index_info


def _extract_transform_chain_info(
    index: faiss.IndexPreTransform,
) -> List[Dict[str, Any]]:
    """
    Extract information about the transformation chain in a PreTransform index.

    Args:
        index: The PreTransform index

    Returns:
        list: List of transformation information dictionaries
    """
    transform_info = []
    for i in range(index.chain.size()):
        transform = index.chain.at(i)
        transform_type = type(transform).__name__

        t_info = {"type": transform_type}

        if hasattr(transform, "d_in"):
            t_info["input_dim"] = transform.d_in
        if hasattr(transform, "d_out"):
            t_info["output_dim"] = transform.d_out
        if hasattr(transform, "is_trained"):
            t_info["is_trained"] = transform.is_trained

        transform_info.append(t_info)

    return transform_info


def _save_vector_data(
    index: Any, filepath: Path, index_info: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Save vector data from an index if reconstruction is possible.

    Args:
        index: The FAISS index
        filepath: Base filepath for saving
        index_info: Index metadata containing dimension info

    Returns:
        tuple: (success, vectors_path)
    """
    if index.ntotal == 0 or not hasattr(index, "reconstruct"):
        return False, None

    try:
        vectors = np.zeros((index.ntotal, index_info["dimension"]), dtype=np.float32)
        for i in range(index.ntotal):
            vectors[i] = index.reconstruct(i)

        vectors_path = str(filepath) + DEFAULT_VECTORS_EXTENSION
        np.save(vectors_path, vectors)
        logger.debug(f"Saved {index.ntotal} vectors to {vectors_path}")
        return True, vectors_path
    except Exception as e:
        logger.warning(f"Failed to save vectors: {str(e)}")
        return False, None


def save_index(
    index: Any,
    filepath: Union[str, Path],
    save_vectors: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Save a FAISS index to disk with comprehensive metadata and optional vector backup.

    This function provides enterprise-grade index persistence with multiple data formats,
    extensive metadata extraction, and robust error handling. It supports all FAISS index
    types and automatically handles fallback serialization methods when needed.

    Args:
        index: The FAISS index to save (must be a valid FAISS index object)
        filepath: Path where the index should be saved (without extension)
        save_vectors: Whether to also save vector data for reconstruction
        metadata: Additional metadata to save with the index

    Returns:
        dict: Comprehensive information about the saved index files including:
            - index_path: Path to the saved index file
            - metadata_path: Path to the metadata JSON file
            - has_vectors: Whether vector data was successfully saved
            - vectors_path: Path to vectors file (if has_vectors is True)
            - timestamp: Unix timestamp of save operation

    Raises:
        TypeError: If the index is not a valid FAISS index
        OSError: If file operations fail
        RuntimeError: If index serialization fails

    Example:
        >>> index = faiss.IndexFlatL2(128)
        >>> result = save_index(index, "/path/to/my_index", save_vectors=True)
        >>> print(f"Index saved to: {result['index_path']}")
    """
    # Validate input parameters
    _validate_index(index)
    filepath = Path(filepath)

    # Ensure output directory exists
    _create_directory_if_needed(filepath)

    # Create directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Prepare result information
    result = {
        "index_path": str(filepath) + DEFAULT_INDEX_EXTENSION,
        "metadata_path": str(filepath) + DEFAULT_METADATA_EXTENSION,
        "has_vectors": False,
        "timestamp": time.time(),
    }

    # Gather index information
    index_info = {
        "index_type": type(index).__name__,
        "dimension": getattr(index, "d", None),
        "ntotal": index.ntotal,
        "is_trained": getattr(index, "is_trained", True),
        "timestamp": time.time(),
    }

    # Add specialized index parameters
    if isinstance(index, faiss.IndexIVF):
        index_info.update(
            {
                "nlist": index.nlist,
                "nprobe": index.nprobe,
                "quantizer_type": type(index.quantizer).__name__,
            }
        )
    elif isinstance(index, faiss.IndexHNSW):
        index_info.update(
            {
                "hnsw_m": index.hnsw.M,
                "ef_search": index.hnsw.efSearch,
                "ef_construction": index.hnsw.efConstruction,
            }
        )
    elif isinstance(index, faiss.IndexPQ):
        index_info.update(
            {
                "pq_m": index.pq.M,
                "pq_nbits": index.pq.nbits,
            }
        )
    elif isinstance(index, faiss.IndexBinaryFlat):
        index_info.update({"is_binary": True, "dimension_bits": index.d})
    elif isinstance(index, faiss.IndexPreTransform):
        # For IndexPreTransform, store information about the transformation chain
        transform_info = []
        for i in range(index.chain.size()):
            transform = index.chain.at(i)
            transform_type = type(transform).__name__

            t_info = {"type": transform_type}

            if hasattr(transform, "d_in"):
                t_info["input_dim"] = transform.d_in
            if hasattr(transform, "d_out"):
                t_info["output_dim"] = transform.d_out
            if hasattr(transform, "is_trained"):
                t_info["is_trained"] = transform.is_trained

            transform_info.append(t_info)

        index_info["transforms"] = transform_info
        index_info["base_index_type"] = type(index.index).__name__

    # Merge with provided metadata
    if metadata:
        index_info.update(metadata)

    # Save metadata
    with open(str(filepath) + DEFAULT_METADATA_EXTENSION, "w") as f:
        json.dump(index_info, f, indent=2)

    # Save vector data if requested and possible
    if save_vectors and index.ntotal > 0 and hasattr(index, "reconstruct"):
        try:
            vectors = np.zeros(
                (index.ntotal, index_info["dimension"]), dtype=np.float32
            )
            for i in range(index.ntotal):
                vectors[i] = index.reconstruct(i)

            np.save(str(filepath) + DEFAULT_VECTORS_EXTENSION, vectors)
            result["has_vectors"] = True
            result["vectors_path"] = str(filepath) + DEFAULT_VECTORS_EXTENSION
        except Exception as e:
            # If reconstruction fails, just log it and continue
            logger.warning(f"Warning: Failed to save vectors: {str(e)}")

    # Save the index
    try:
        faiss.write_index(index, str(filepath) + DEFAULT_INDEX_EXTENSION)
    except RuntimeError as e:
        # Handle special cases where direct write doesn't work
        if "SWIG director method error" in str(e) or "Cannot serialize" in str(e):
            # Use pickle for indices that FAISS can't directly serialize
            with open(str(filepath) + DEFAULT_INDEX_EXTENSION, "wb") as f:
                pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise

    return result


def load_index(
    filepath: Union[str, Path], load_vectors: bool = True
) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a FAISS index from disk with its metadata.

    Args:
        filepath: Path to the index (without extension)
        load_vectors: Whether to also load vector data if available

    Returns:
        tuple: (index, metadata)

    Raises:
        FileNotFoundError: If the index file doesn't exist
        RuntimeError: If the index cannot be loaded
    """
    filepath = Path(filepath)

    # Check if the index file exists
    index_path = str(filepath) + DEFAULT_INDEX_EXTENSION
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")

    # Load metadata if it exists
    metadata_path = str(filepath) + DEFAULT_METADATA_EXTENSION
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    # Load the index
    try:
        index = faiss.read_index(index_path)
    except RuntimeError as e:
        # Try pickle if FAISS reading fails
        if "RuntimeError" in str(e):
            try:
                with open(index_path, "rb") as f:
                    index = pickle.load(f)
            except Exception as sub_e:
                raise RuntimeError(
                    f"Failed to load index: {str(e)}, pickle error: {str(sub_e)}"
                )
        else:
            raise

    # Load vectors if requested and available
    vectors_path = str(filepath) + DEFAULT_VECTORS_EXTENSION
    if load_vectors and os.path.exists(vectors_path):
        try:
            metadata["vectors_loaded"] = True
            metadata["vectors_path"] = vectors_path
        except Exception as e:
            metadata["vectors_loaded"] = False
            metadata["vectors_load_error"] = str(e)

    return index, metadata


def extract_parameters(index: Any) -> Dict[str, Any]:
    """
    Extract parameters from a FAISS index for transfer or serialization.

    Args:
        index: The FAISS index to extract parameters from

    Returns:
        dict: Extracted parameters
    """
    params = {
        "index_type": type(index).__name__,
        "dimension": getattr(index, "d", None),
        "ntotal": index.ntotal,
        "is_trained": getattr(index, "is_trained", True),
    }

    # Extract specialized parameters by index type
    if isinstance(index, faiss.IndexIVF):
        params.update(
            {
                "nlist": index.nlist,
                "nprobe": index.nprobe,
                "metric_type": index.metric_type,
                "direct_map": {
                    "type": index.direct_map.type,
                    "is_trained": index.is_trained,
                },
            }
        )

        # Extract quantizer parameters
        q_params = extract_parameters(index.quantizer)
        params["quantizer"] = q_params

    elif isinstance(index, faiss.IndexHNSW):
        params.update(
            {
                "hnsw": {
                    "M": index.hnsw.M,
                    "efConstruction": index.hnsw.efConstruction,
                    "efSearch": index.hnsw.efSearch,
                    "level_generator_seed": index.hnsw.level_generator_seed,
                    "upper_beam": index.hnsw.upper_beam,
                },
                "metric_type": index.metric_type,
            }
        )

    elif isinstance(index, faiss.IndexPQ):
        params.update(
            {
                "pq": {
                    "M": index.pq.M,
                    "nbits": index.pq.nbits,
                    "centroids": (
                        index.pq.centroids.shape
                        if hasattr(index.pq, "centroids")
                        else None
                    ),
                },
                "metric_type": index.metric_type,
            }
        )

    elif isinstance(index, faiss.IndexBinaryFlat):
        params.update({"is_binary": True, "dimension_bits": index.d})

    elif isinstance(index, faiss.IndexPreTransform):
        # For IndexPreTransform, extract transform chain
        transforms = []
        for i in range(index.chain.size()):
            transform = index.chain.at(i)
            t_type = type(transform).__name__

            t_params = {"type": t_type}

            if hasattr(transform, "d_in"):
                t_params["d_in"] = transform.d_in
            if hasattr(transform, "d_out"):
                t_params["d_out"] = transform.d_out
            if hasattr(transform, "is_trained"):
                t_params["is_trained"] = transform.is_trained

            transforms.append(t_params)

        params["transforms"] = transforms

        # Extract base index parameters
        base_params = extract_parameters(index.index)
        params["base_index"] = base_params

    return params


def serialize_index(
    index: Any, include_vectors: bool = False, max_vectors: int = 1000
) -> bytes:
    """
    Serialize a FAISS index to a binary format for transfer.

    Args:
        index: The FAISS index to serialize
        include_vectors: Whether to include vector data
        max_vectors: Maximum number of vectors to include

    Returns:
        bytes: Serialized index data
    """
    # Create a temporary file for the index
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Write the index to the temporary file
        faiss.write_index(index, tmp_path)

        # Read the serialized data
        with open(tmp_path, "rb") as f:
            index_data = f.read()

        # If vectors should be included, extract them
        vectors_data = None
        if include_vectors and index.ntotal > 0 and hasattr(index, "reconstruct"):
            n_vectors = min(index.ntotal, max_vectors)

            try:
                dimension = getattr(index, "d", None)
                if dimension is None:
                    # Try to get dimension from the first vector
                    dimension = len(index.reconstruct(0))

                vectors = np.zeros((n_vectors, dimension), dtype=np.float32)
                for i in range(n_vectors):
                    vectors[i] = index.reconstruct(i)

                # Serialize the vectors
                with tempfile.NamedTemporaryFile(delete=False) as tmp_vec:
                    np.save(tmp_vec, vectors)
                    tmp_vec_path = tmp_vec.name

                with open(tmp_vec_path, "rb") as f:
                    vectors_data = f.read()

                # Clean up
                os.unlink(tmp_vec_path)
            except Exception as e:
                logger.warning(f"Warning: Failed to serialize vectors: {str(e)}")

        # Extract parameters
        params = extract_parameters(index)

        # Create the serialized package
        serialized = {
            "index_data": index_data,
            "parameters": params,
            "vectors_data": vectors_data,
            "include_vectors": include_vectors,
            "timestamp": time.time(),
        }

        # Return the pickled data
        return pickle.dumps(serialized, protocol=pickle.HIGHEST_PROTOCOL)

    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def deserialize_index(data: bytes) -> Tuple[Any, Dict[str, Any]]:
    """
    Deserialize a FAISS index from binary data.

    Args:
        data: Serialized index data

    Returns:
        tuple: (index, metadata)

    Raises:
        ValueError: If the data is invalid
    """
    try:
        # Unpickle the serialized package
        serialized = pickle.loads(data)

        # Extract components
        index_data = serialized.get("index_data")
        params = serialized.get("parameters", {})
        vectors_data = serialized.get("vectors_data")

        if not index_data:
            raise ValueError("Invalid serialized data: missing index_data")

        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(index_data)
            tmp_path = tmp.name

        # Load the index
        try:
            index = faiss.read_index(tmp_path)
        except Exception as e:
            raise ValueError(f"Failed to deserialize index: {str(e)}")
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        # Load vectors if included
        if vectors_data:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_vec:
                tmp_vec.write(vectors_data)
                tmp_vec_path = tmp_vec.name

            try:
                vectors = np.load(tmp_vec_path)
                params["vectors"] = vectors
            except Exception as e:
                params["vectors_error"] = str(e)
            finally:
                # Clean up
                if os.path.exists(tmp_vec_path):
                    os.unlink(tmp_vec_path)

        return index, params

    except Exception as e:
        raise ValueError(f"Failed to deserialize index: {str(e)}")


def export_trained_parameters(index: Any) -> Dict[str, Any]:
    """
    Export trained parameters from an index for reuse.

    This is useful for creating new indices with the same training.

    Args:
        index: The trained FAISS index

    Returns:
        dict: Trained parameters
    """
    params = {}

    # Export parameters based on index type
    if isinstance(index, faiss.IndexIVF):
        if not index.is_trained:
            return {"error": "Index is not trained"}

        # Extract coarse quantizer centroids
        if isinstance(index.quantizer, faiss.IndexFlat):
            quantizer_vectors = index.quantizer.xb
            if quantizer_vectors is not None and quantizer_vectors.size > 0:
                params["coarse_centroids"] = np.array(quantizer_vectors)
                params["nlist"] = index.nlist

    elif isinstance(index, faiss.IndexPQ):
        if not index.is_trained:
            return {"error": "Index is not trained"}

        # Extract PQ centroids
        if hasattr(index.pq, "centroids"):
            params["pq_centroids"] = np.array(index.pq.centroids)
            params["pq_M"] = index.pq.M
            params["pq_nbits"] = index.pq.nbits

    elif isinstance(index, faiss.IndexPreTransform):
        # Extract transform parameters
        transform_params = []
        for i in range(index.chain.size()):
            transform = index.chain.at(i)
            t_type = type(transform).__name__
            t_param = {"type": t_type}

            # PCA matrix
            if isinstance(transform, faiss.PCAMatrix):
                if hasattr(transform, "eigen_vectors"):
                    t_param["eigen_vectors"] = np.array(transform.eigen_vectors)
                if hasattr(transform, "eigen_values"):
                    t_param["eigen_values"] = np.array(transform.eigen_values)
                if hasattr(transform, "mean"):
                    t_param["mean"] = np.array(transform.mean)
                t_param["d_in"] = transform.d_in
                t_param["d_out"] = transform.d_out

            # OPQ matrix
            elif isinstance(transform, faiss.OPQMatrix):
                if hasattr(transform, "A"):
                    t_param["A"] = np.array(transform.A)
                t_param["d_in"] = transform.d_in
                t_param["d_out"] = transform.d_out
                t_param["M"] = transform.M

            transform_params.append(t_param)

        params["transforms"] = transform_params

        # Also get base index parameters
        base_params = export_trained_parameters(index.index)
        params["base_index"] = base_params

    return params
