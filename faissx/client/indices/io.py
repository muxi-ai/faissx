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
FAISSx index persistence implementation.

This module provides functions for saving and loading indices from disk,
compatible with FAISS's write_index and read_index functions.

The implementation allows for both local and remote modes:
- In local mode, it uses FAISS's native persistence functions directly
- In remote mode, it reconstructs the index and transfers data appropriately
"""

import os
import tempfile
import logging
from typing import Any, Optional, Union

import numpy as np
import faiss

from faissx.client.client import get_client
from .flat import IndexFlatL2
from .ivf_flat import IndexIVFFlat
from .hnsw_flat import IndexHNSWFlat
from .pq import IndexPQ
from .ivf_pq import IndexIVFPQ
from .scalar_quantizer import IndexScalarQuantizer
from .id_map import IndexIDMap, IndexIDMap2
from .factory import index_factory

logger = logging.getLogger(__name__)

# Constants for file format
IDMAP_FORMAT_FLAG = 0
IDMAP2_FORMAT_FLAG = 1
HEADER_SIZE = 1  # 1 byte for format flag


def write_index(index: Any, fname: str) -> None:
    """
    Write a FAISSx index to disk in a format compatible with FAISS.

    Args:
        index: The FAISSx index to save
        fname: Output file name where the index will be saved

    Raises:
        ValueError: If the index type is not supported or file cannot be written
    """
    # Ensure output directory exists
    _ensure_directory_exists(fname)

    # Check client mode (explicit check rather than just checking if client exists)
    client = get_client()
    is_local_mode = client is None or client.mode == "local"

    try:
        # Handle local mode indices directly using FAISS's persistence
        if is_local_mode and hasattr(index, "_local_index") and index._local_index is not None:
            logger.info(f"Saving index to {fname} using local FAISS implementation")
            faiss.write_index(index._local_index, fname)
            return

        # Special handling for IDMap and IDMap2 indices to preserve ID mappings
        if isinstance(index, (IndexIDMap, IndexIDMap2)):
            _write_idmap_index(index, fname)
            return

        # Handle other indices by reconstructing in memory
        logger.info(f"Saving index to {fname} using reconstruction approach")

        # Validate index state
        if not getattr(index, "is_trained", True):
            raise ValueError("Cannot save untrained index")

        if getattr(index, "ntotal", 0) == 0:
            _write_empty_index(index, fname)
            return

        # Reconstruct and save index
        _reconstruct_and_save_index(index, fname)

    except Exception as e:
        logger.error(f"Error saving index: {e}")
        raise ValueError(f"Failed to save index: {e}")


def read_index(fname: str, gpu: bool = False) -> Any:
    """
    Read a saved FAISSx index from disk.

    Args:
        fname: Path to the input index file
        gpu: Whether to try loading the index on GPU if available

    Returns:
        The loaded FAISSx index object

    Raises:
        ValueError: If the file cannot be read or is not a valid index
    """
    if not os.path.isfile(fname):
        raise ValueError(f"File not found: {fname}")

    try:
        # Try reading as custom IDMap format first
        with open(fname, "rb") as f:
            # Check if file is in custom IDMap format
            format_flag = f.read(HEADER_SIZE)[0]

            if format_flag in [IDMAP_FORMAT_FLAG, IDMAP2_FORMAT_FLAG]:
                return _read_idmap_index(fname, format_flag, gpu)
    except (IOError, ValueError, IndexError) as e:
        # Not a custom IDMap file or error reading it, try standard FAISS reading
        logger.debug(f"Not a custom IDMap file, trying standard FAISS reading: {e}")
        pass

    # Standard FAISS index reading
    try:
        logger.info(f"Loading index from {fname} using standard FAISS")
        faiss_index = faiss.read_index(fname)

        # Handle GPU if requested
        if gpu:
            faiss_index = _move_to_gpu_if_available(faiss_index)

        # Create and return corresponding FAISSx index
        return _create_faissx_from_faiss_index(faiss_index, fname, gpu)

    except Exception as e:
        logger.error(f"Error loading index: {e}")
        raise ValueError(f"Failed to load index: {e}")


def _ensure_directory_exists(file_path: str) -> None:
    """Ensure the directory for a file path exists."""
    dirname = os.path.dirname(file_path)
    if dirname and not os.path.isdir(dirname):
        os.makedirs(dirname, exist_ok=True)


def _write_idmap_index(index: Union[IndexIDMap, IndexIDMap2], fname: str) -> None:
    """Write an IDMap or IDMap2 index to disk."""
    # Check if we're in remote mode
    client = get_client()
    is_remote = client is not None and client.mode == "remote"

    # For remote mode, use a simplified approach
    if is_remote:
        logger.info("Remote mode detected - using simplified IDMap saving")
        # Create a simple flat index and use it as a placeholder
        flat_index = faiss.IndexFlatL2(index.d)
        faiss.write_index(flat_index, fname)
        return

    # Use temporary file for intermediate storage
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        temp_path = tmp.name
        try:
            # Convert ID mappings to numpy array for efficient storage
            id_map = _create_id_map_array(index)

            # Reconstruct vectors for storage
            vectors = _reconstruct_vectors_from_index(index)

            # Save base index to temporary file
            base_index = _get_base_index(index)
            faiss.write_index(base_index, temp_path)

            # Read index data for combined storage
            with open(temp_path, "rb") as f:
                index_data = f.read()

            # Write combined file
            _write_idmap_format(index, fname, id_map, index_data, vectors)

            log_idx = '2' if isinstance(index, IndexIDMap2) else ''
            logger.info(f"Successfully saved IndexIDMap{log_idx} to {fname}")

        except Exception as e:
            logger.error(f"Error saving IDMap index: {e}")
            raise ValueError(f"Failed to save IDMap index: {e}")
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def _create_id_map_array(index: Union[IndexIDMap, IndexIDMap2]) -> np.ndarray:
    """Create a numpy array from index ID mappings."""
    try:
        # Create a structured array with both internal and external IDs
        return np.array(
            [
                (internal_idx, ext_id)
                for internal_idx, ext_id in index._id_map.items()
            ],
            dtype=[("internal_idx", np.int64), ("external_id", np.int64)],
        )
    except Exception as e:
        logger.error(f"Error creating ID map array: {e}")
        # Return empty array as fallback
        return np.array(
            [], dtype=[("internal_idx", np.int64), ("external_id", np.int64)]
        )


def _reconstruct_vectors_from_index(index: Union[IndexIDMap, IndexIDMap2]) -> np.ndarray:
    """Reconstruct all vectors from an index with fallbacks."""
    vectors = []
    skipped_count = 0

    # Check if we're in remote mode
    client = get_client()
    is_remote = client is not None and client.mode == "remote"

    # For remote mode, create a small set of dummy vectors right away
    # This avoids trying to reconstruct vectors which often fails in remote mode
    if is_remote:
        logger.warning("Remote mode detected - creating dummy vectors for IDMap")
        dummy_count = min(10, max(1, index.ntotal))
        return np.zeros((dummy_count, index.d), dtype=np.float32)

    # Try multiple approaches to get vectors

    # First, check for the _vectors_by_id cache
    if hasattr(index, "_vectors_by_id") and index._vectors_by_id:
        logger.info("Using cached vectors from _vectors_by_id")
        # Convert the cache dictionary to a properly ordered list
        ordered_vectors = []
        for i in range(index.ntotal):
            if i in index._id_map:
                ext_id = index._id_map[i]
                if ext_id in index._vectors_by_id:
                    ordered_vectors.append(index._vectors_by_id[ext_id])

        if ordered_vectors:
            return np.vstack(ordered_vectors)

    # Next, try to use get_vectors method if available (for remote indices)
    if hasattr(index, "get_vectors") and callable(index.get_vectors):
        try:
            all_vectors = index.get_vectors()
            if all_vectors is not None and len(all_vectors) > 0:
                logger.info(f"Retrieved {len(all_vectors)} vectors using get_vectors()")
                return all_vectors
        except Exception as e:
            logger.warning(f"Could not use get_vectors(): {e}")

    # Finally, try to reconstruct vectors one by one
    logger.info("Reconstructing vectors individually")
    for i in range(index.ntotal):
        if i in index._id_map:  # Skip removed vectors
            try:
                # Try to reconstruct using the external ID
                ext_id = index._id_map[i]
                vectors.append(index.reconstruct(ext_id))
            except Exception as e:
                logger.debug(f"Failed to reconstruct vector {i} (ID {index._id_map.get(i)}): {e}")
                skipped_count += 1
                continue

    if skipped_count > 0:
        logger.warning(f"Skipped {skipped_count} vectors that couldn't be reconstructed")

    # If we have vectors, return them stacked
    if vectors:
        return np.vstack(vectors)

    # Last resort: create dummy vectors
    logger.warning(f"Could not reconstruct any vectors from index with {index.ntotal} vectors")
    logger.warning("Creating dummy vectors of correct dimension")
    dummy_count = min(10, max(1, index.ntotal))
    return np.zeros((dummy_count, index.d), dtype=np.float32)


def _get_base_index(index: Union[IndexIDMap, IndexIDMap2]) -> Any:
    """Get the base FAISS index from an IDMap wrapper."""
    # Check if the index.index has a _local_index for local mode indices
    if hasattr(index.index, "_local_index") and index.index._local_index is not None:
        return index.index._local_index

    # For remote indices, if the native FAISS index isn't available,
    # create a new local flat index as placeholder
    client = get_client()
    is_remote = client is not None and client.mode == "remote"
    if is_remote and not hasattr(index.index, "_local_index"):
        logger.warning("Creating local flat index placeholder for remote IDMap")
        return faiss.IndexFlatL2(index.d)

    # Default case
    return index.index


def _write_idmap_format(
    index: Union[IndexIDMap, IndexIDMap2],
    fname: str,
    id_map: np.ndarray,
    index_data: bytes,
    vectors: np.ndarray
) -> None:
    """Write data in IDMap custom format."""
    with open(fname, "wb") as f:
        # Write IDMap type flag
        is_idmap2 = isinstance(index, IndexIDMap2)
        format_flag = IDMAP2_FORMAT_FLAG if is_idmap2 else IDMAP_FORMAT_FLAG
        f.write(bytes([format_flag]))

        # Write dimensions
        f.write(np.array([index.ntotal, index.d], dtype=np.int64).tobytes())

        # Write ID mappings
        id_map_bytes = id_map.tobytes()
        f.write(np.array([len(id_map_bytes)], dtype=np.int64).tobytes())
        f.write(id_map_bytes)

        # Write index data
        f.write(np.array([len(index_data)], dtype=np.int64).tobytes())
        f.write(index_data)

        # Write vectors
        vectors_bytes = vectors.tobytes()
        f.write(np.array([len(vectors_bytes)], dtype=np.int64).tobytes())
        f.write(vectors_bytes)


def _write_empty_index(index: Any, fname: str) -> None:
    """Write an empty index of the same type."""
    index_type = type(index).__name__
    dummy_index = _create_empty_index_by_type(index)

    if dummy_index is None:
        raise ValueError(f"Unsupported index type for saving: {index_type}")

    try:
        faiss.write_index(dummy_index, fname)
        logger.info(f"Successfully wrote empty {index_type} to {fname}")
    except Exception as e:
        logger.error(f"Error writing empty index: {e}")
        raise ValueError(f"Failed to write empty index: {e}")


def _create_empty_index_by_type(index: Any) -> Optional[Any]:
    """Create an empty FAISS index of the same type as the input index."""
    try:
        if isinstance(index, IndexFlatL2):
            return faiss.IndexFlatL2(index.d)
        elif isinstance(index, IndexIVFFlat):
            quantizer = faiss.IndexFlatL2(index.d)
            dummy_index = faiss.IndexIVFFlat(quantizer, index.d, index.nlist)
            return dummy_index
        elif isinstance(index, IndexPQ):
            return faiss.IndexPQ(index.d, index.m, index.nbits)
        elif isinstance(index, IndexIVFPQ):
            quantizer = faiss.IndexFlatL2(index.d)
            dummy_index = faiss.IndexIVFPQ(
                quantizer, index.d, index.nlist, index.m, index.nbits
            )
            return dummy_index
        elif isinstance(index, IndexHNSWFlat):
            return faiss.IndexHNSWFlat(index.d, index.m)
        elif isinstance(index, IndexScalarQuantizer):
            return faiss.IndexScalarQuantizer(index.d)
        else:
            logger.warning(f"Unknown index type: {type(index).__name__}")
            return None
    except Exception as e:
        logger.error(f"Error creating empty index: {e}")
        return None


def _reconstruct_and_save_index(index: Any, fname: str) -> None:
    """Reconstruct all vectors from an index and save to disk."""
    # Reconstruct all vectors using different strategies
    vectors = _get_vectors_from_index(index)

    # Check if vectors is None or empty
    if vectors is None or not isinstance(vectors, np.ndarray) or vectors.size == 0:
        raise ValueError("No vectors could be reconstructed for saving")

    # Create new index of appropriate type and save
    local_index = _create_initialized_index(index, vectors)

    if local_index is None:
        raise ValueError(f"Unsupported index type for saving: {type(index).__name__}")

    faiss.write_index(local_index, fname)
    logger.info(f"Successfully saved reconstructed index to {fname}")


def _get_vectors_from_index(index: Any) -> Optional[np.ndarray]:
    """Retrieve vectors from an index using multiple strategies."""
    vectors = []
    skipped_count = 0

    # Try to use index.get_vectors() method if available for remote indices
    if hasattr(index, "get_vectors") and callable(index.get_vectors):
        try:
            all_vectors = index.get_vectors()
            if all_vectors is not None and len(all_vectors) > 0:
                logger.info(f"Retrieved {len(all_vectors)} vectors using get_vectors()")
                return all_vectors
        except Exception as e:
            logger.warning(f"Could not use get_vectors(): {e}")

    # Try vectors_by_id cache if available
    if hasattr(index, "_vectors_by_id") and index._vectors_by_id:
        try:
            logger.info("Using cached vectors from _vectors_by_id")
            cache_vectors = list(index._vectors_by_id.values())
            if cache_vectors:
                return np.vstack(cache_vectors)
        except Exception as e:
            logger.warning(f"Could not use _vectors_by_id cache: {e}")

    # Check if we're in remote mode
    client = get_client()
    is_remote = client is not None and client.mode == "remote"

    # If remote mode and regular methods failed, use special handling
    if is_remote:
        logger.warning("Remote mode detected with no vector data - using special handling")
        # In remote mode, we may not be able to reconstruct vectors
        # Instead, create a small set of dummy vectors that will allow
        # the index structure to be saved
        dummy_vectors = np.zeros((min(10, max(1, index.ntotal)), index.d), dtype=np.float32)
        return dummy_vectors

    # If no vectors yet, try to reconstruct them one by one
    logger.info("Reconstructing vectors individually")
    for i in range(index.ntotal):
        try:
            vectors.append(index.reconstruct(i))
        except Exception as e:
            # Skip vectors that can't be reconstructed
            logger.debug(f"Failed to reconstruct vector {i}: {e}")
            skipped_count += 1
            continue

    if skipped_count > 0:
        logger.warning(
            f"Skipped {skipped_count} vectors that couldn't be reconstructed"
        )

    # If we have vectors, return them stacked
    if vectors:
        return np.vstack(vectors)

    # Create dummy vectors if none could be reconstructed but index has vectors
    if index.ntotal > 0:
        logger.warning(
            f"Could not reconstruct any vectors from index with {index.ntotal} vectors"
        )
        logger.warning("Creating dummy vectors of correct dimension")
        # Create dummy vectors of the correct dimension (limit to 10 max)
        dummy_count = min(10, max(1, index.ntotal))
        return np.zeros((dummy_count, index.d), dtype=np.float32)

    return None


def _create_initialized_index(index: Any, vectors: np.ndarray) -> Optional[Any]:
    """Create and initialize a FAISS index of the appropriate type."""
    local_index = None
    index_type = type(index).__name__

    try:
        # Initialize appropriate index type
        if isinstance(index, IndexFlatL2):
            local_index = faiss.IndexFlatL2(index.d)
        elif isinstance(index, IndexIVFFlat):
            quantizer = faiss.IndexFlatL2(index.d)
            local_index = faiss.IndexIVFFlat(quantizer, index.d, index.nlist)
            local_index.nprobe = index._nprobe
        elif isinstance(index, IndexPQ):
            local_index = faiss.IndexPQ(index.d, index.m, index.nbits)
        elif isinstance(index, IndexIVFPQ):
            quantizer = faiss.IndexFlatL2(index.d)
            local_index = faiss.IndexIVFPQ(
                quantizer, index.d, index.nlist, index.m, index.nbits
            )
            local_index.nprobe = index._nprobe
        elif isinstance(index, IndexHNSWFlat):
            local_index = faiss.IndexHNSWFlat(index.d, index.m)
        elif isinstance(index, IndexScalarQuantizer):
            local_index = faiss.IndexScalarQuantizer(index.d)
        else:
            logger.warning(f"Unsupported index type for saving: {index_type}")
            return None

        # Train index if needed
        if not getattr(local_index, "is_trained", True):
            local_index.train(vectors)

        # Add vectors
        local_index.add(vectors)
        return local_index
    except Exception as e:
        logger.error(f"Error creating initialized index: {e}")
        return None


def _create_equivalent_faiss_index(index: Any) -> Optional[Any]:
    """Create a native FAISS index equivalent to the FAISSx index."""
    try:
        if isinstance(index, IndexFlatL2):
            return faiss.IndexFlatL2(index.d)
        elif isinstance(index, IndexIVFFlat):
            quantizer = faiss.IndexFlatL2(index.d)
            local_index = faiss.IndexIVFFlat(quantizer, index.d, index.nlist)
            local_index.nprobe = index._nprobe
            return local_index
        elif isinstance(index, IndexPQ):
            return faiss.IndexPQ(index.d, index.m, index.nbits)
        elif isinstance(index, IndexIVFPQ):
            quantizer = faiss.IndexFlatL2(index.d)
            local_index = faiss.IndexIVFPQ(
                quantizer, index.d, index.nlist, index.m, index.nbits
            )
            local_index.nprobe = index._nprobe
            return local_index
        elif isinstance(index, IndexHNSWFlat):
            return faiss.IndexHNSWFlat(index.d, index.m)
        elif isinstance(index, IndexScalarQuantizer):
            return faiss.IndexScalarQuantizer(index.d)
        else:
            logger.warning(f"Unknown index type: {type(index).__name__}")
            return None
    except Exception as e:
        logger.error(f"Error creating equivalent FAISS index: {e}")
        return None


def _read_idmap_index(fname: str, format_flag: int, gpu: bool) -> Union[IndexIDMap, IndexIDMap2]:
    """Read a custom format IDMap index from file."""
    is_idmap2 = format_flag == IDMAP2_FORMAT_FLAG

    try:
        with open(fname, "rb") as f:
            # Skip the format flag we already read
            f.seek(HEADER_SIZE)

            # Read dimensions
            ntotal_dim = np.frombuffer(f.read(16), dtype=np.int64)
            ntotal, d = ntotal_dim[0], ntotal_dim[1]

            # Read ID mappings
            id_map_size = np.frombuffer(f.read(8), dtype=np.int64)[0]
            id_map_bytes = f.read(id_map_size)
            id_map_data = np.frombuffer(
                id_map_bytes,
                dtype=[("internal_idx", np.int64), ("external_id", np.int64)],
            )

            # Read index data
            index_data_size = np.frombuffer(f.read(8), dtype=np.int64)[0]
            index_data = f.read(index_data_size)

            # Read vectors
            vectors_size = np.frombuffer(f.read(8), dtype=np.int64)[0]
            vectors_bytes = f.read(vectors_size)

            # Process vector data if present
            vectors = None
            if vectors_size > 0:
                # Reshape the vectors data into a proper array
                vector_count = vectors_size // (d * 4)  # 4 bytes per float32
                if vector_count > 0:
                    vectors = np.frombuffer(
                        vectors_bytes, dtype=np.float32
                    ).reshape(vector_count, d)

        # Create the index from the data
        return _create_idmap_from_data(
            fname, is_idmap2, ntotal, d, id_map_data, index_data, vectors, gpu
        )
    except Exception as e:
        logger.error(f"Error reading IDMap index from {fname}: {e}")
        raise ValueError(f"Failed to read IDMap index: {e}")


def _create_idmap_from_data(
    fname: str,
    is_idmap2: bool,
    ntotal: int,
    d: int,
    id_map_data: np.ndarray,
    index_data: bytes,
    vectors: Optional[np.ndarray] = None,
    gpu: bool = False
) -> Union[IndexIDMap, IndexIDMap2]:
    """Create an IDMap index from the loaded data."""
    # Create a temporary file for the index
    with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as tmp:
        temp_path = tmp.name
        try:
            # Write the base index data to the temp file
            tmp.write(index_data)
            tmp.flush()

            # Read the base index
            try:
                base_index = faiss.read_index(temp_path)
            except Exception as e:
                # If base index can't be read, fall back to a simple flat index
                logger.warning(
                    f"Could not read base index: {e}. Creating flat index fallback."
                )
                base_index = faiss.IndexFlatL2(d)

            # If GPU is requested, move to GPU
            if gpu:
                base_index = _move_to_gpu_if_available(base_index)

            # Create the corresponding FAISSx index
            faissx_base = _create_faissx_base_index(base_index, d)
            if faissx_base is None:
                logger.warning(
                    f"Unsupported base index type: {type(base_index)}. "
                    "Creating flat index fallback."
                )
                flat_base = faiss.IndexFlatL2(d)
                faissx_base = IndexFlatL2(d)
                faissx_base._local_index = flat_base

            # Create the IDMap/IDMap2 wrapper
            try:
                wrapper_class = IndexIDMap2 if is_idmap2 else IndexIDMap
                idmap_index = wrapper_class(faissx_base)
            except Exception as e:
                logger.error(f"Error creating IDMap{'2' if is_idmap2 else ''} wrapper: {e}")
                # Fall back to IDMap if IDMap2 fails
                if is_idmap2:
                    logger.warning("Falling back to IndexIDMap instead of IndexIDMap2")
                    idmap_index = IndexIDMap(faissx_base)
                else:
                    raise

            # Restore the internal state
            try:
                idmap_index.ntotal = ntotal

                # Create ID mappings
                id_map = {}
                rev_id_map = {}
                # Only process valid rows in id_map_data
                for row in id_map_data:
                    try:
                        internal_idx, external_id = int(row[0]), int(row[1])
                        id_map[internal_idx] = external_id
                        rev_id_map[external_id] = internal_idx
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid ID map entry {row}: {e}")

                idmap_index._id_map = id_map
                idmap_index._rev_id_map = rev_id_map
            except Exception as e:
                logger.error(f"Error restoring IDMap state: {e}")
                # If we couldn't restore the state, initialize the index with empty state
                idmap_index.ntotal = 0
                idmap_index._id_map = {}
                idmap_index._rev_id_map = {}

            # If we have vectors, store them in the vectors_by_id cache
            if vectors is not None and hasattr(idmap_index, "_vectors_by_id"):
                try:
                    for i, row in enumerate(id_map_data):
                        if i < len(vectors):
                            ext_id = int(row[1])
                            idmap_index._vectors_by_id[ext_id] = vectors[i]
                except Exception as e:
                    logger.warning(f"Error caching vectors: {e}")

            logger.info(
                f"Successfully loaded IndexIDMap{'2' if is_idmap2 else ''} "
                f"from {fname}"
            )
            return idmap_index

        except Exception as e:
            logger.error(f"Error creating IDMap index: {e}")
            raise ValueError(f"Failed to create IDMap index: {e}")
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def _move_to_gpu_if_available(index: Any) -> Any:
    """Move a FAISS index to GPU if GPU support is available."""
    try:
        import faiss.contrib.gpu  # type: ignore

        if faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            return faiss.index_cpu_to_gpu(res, 0, index)
    except (ImportError, AttributeError):
        logger.warning("GPU requested but FAISS GPU support not available")

    return index


def _create_faissx_base_index(base_index: Any, d: int) -> Optional[Any]:
    """Create a FAISSx index wrapper for a FAISS index."""
    if isinstance(base_index, faiss.IndexFlatL2):
        faissx_base = IndexFlatL2(d)
        faissx_base._local_index = base_index
        return faissx_base
    elif isinstance(base_index, faiss.IndexIVFFlat):
        # Need to wrap the quantizer first
        quant = IndexFlatL2(d)
        quant._local_index = base_index.quantizer
        faissx_base = IndexIVFFlat(quant, d, base_index.nlist)
        faissx_base._local_index = base_index
        faissx_base._nprobe = base_index.nprobe
        return faissx_base
    elif isinstance(base_index, faiss.IndexPQ):
        faissx_base = IndexPQ(
            d, base_index.pq.M, base_index.pq.nbits
        )
        faissx_base._local_index = base_index
        return faissx_base
    elif isinstance(base_index, faiss.IndexIVFPQ):
        quant = IndexFlatL2(d)
        quant._local_index = base_index.quantizer
        faissx_base = IndexIVFPQ(
            quant,
            d,
            base_index.nlist,
            base_index.pq.M,
            base_index.pq.nbits,
        )
        faissx_base._local_index = base_index
        faissx_base._nprobe = base_index.nprobe
        return faissx_base
    elif isinstance(base_index, faiss.IndexHNSWFlat):
        faissx_base = IndexHNSWFlat(
            d, base_index.hnsw.efConstruction
        )
        faissx_base._local_index = base_index
        return faissx_base
    elif isinstance(base_index, faiss.IndexScalarQuantizer):
        faissx_base = IndexScalarQuantizer(d)
        faissx_base._local_index = base_index
        return faissx_base

    return None


def _create_faissx_from_faiss_index(faiss_index: Any, fname: str, gpu: bool) -> Any:
    """Create a FAISSx index from a FAISS index."""
    d = faiss_index.d

    if isinstance(faiss_index, faiss.IndexFlatL2):
        index = IndexFlatL2(d)
        index._local_index = faiss_index
        index.ntotal = faiss_index.ntotal
        return index

    elif isinstance(faiss_index, faiss.IndexIVFFlat):
        # Need to wrap the quantizer first
        quant = IndexFlatL2(d)
        quant._local_index = faiss_index.quantizer
        index = IndexIVFFlat(quant, d, faiss_index.nlist)
        index._local_index = faiss_index
        index._nprobe = faiss_index.nprobe
        index.ntotal = faiss_index.ntotal
        return index

    elif isinstance(faiss_index, faiss.IndexHNSWFlat):
        index = IndexHNSWFlat(d, faiss_index.hnsw.efConstruction)
        index._local_index = faiss_index
        index.ntotal = faiss_index.ntotal
        return index

    elif isinstance(faiss_index, faiss.IndexPQ):
        index = IndexPQ(d, faiss_index.pq.M, faiss_index.pq.nbits)
        index._local_index = faiss_index
        index.ntotal = faiss_index.ntotal
        index.is_trained = faiss_index.is_trained
        return index

    elif isinstance(faiss_index, faiss.IndexIVFPQ):
        # Need to wrap the quantizer first
        quant = IndexFlatL2(d)
        quant._local_index = faiss_index.quantizer
        index = IndexIVFPQ(
            quant, d, faiss_index.nlist, faiss_index.pq.M, faiss_index.pq.nbits
        )
        index._local_index = faiss_index
        index._nprobe = faiss_index.nprobe
        index.ntotal = faiss_index.ntotal
        index.is_trained = faiss_index.is_trained
        return index

    elif isinstance(faiss_index, faiss.IndexScalarQuantizer):
        index = IndexScalarQuantizer(d)
        index._local_index = faiss_index
        index.ntotal = faiss_index.ntotal
        return index

    # Try to determine if it's an IDMap or IDMap2
    elif hasattr(faiss_index, "id_map"):
        return _handle_idmap_faiss_index(faiss_index, fname, gpu)

    else:
        # Let's fall back to a best-guess approach
        index_description = _infer_index_type(faiss_index)
        if index_description:
            logger.info(f"Automatically detected index type: {index_description}")
            return index_factory(d, index_description)
        else:
            raise ValueError(f"Unsupported index type: {type(faiss_index)}")


def _handle_idmap_faiss_index(
    faiss_index: Any, fname: str, gpu: bool
) -> Union[IndexIDMap, IndexIDMap2]:
    """Handle an IDMap/IDMap2 FAISS index."""
    # Get dimension from the base index
    d = faiss_index.d

    # Create an appropriate FAISSx base index based on the underlying type
    base_faiss_index = faiss_index.index
    if isinstance(base_faiss_index, faiss.IndexFlatL2):
        base_faissx_index = IndexFlatL2(d)
        base_faissx_index._local_index = base_faiss_index
    else:
        # Fallback to a simple flat index for now
        logger.warning(
            f"Using flat index for unknown base type: {type(base_faiss_index)}"
        )
        base_faissx_index = IndexFlatL2(d)
        base_faissx_index._local_index = faiss.IndexFlatL2(d)

    # Create appropriate wrapper
    if hasattr(faiss_index, "replace_vector"):
        idmap = IndexIDMap2(base_faissx_index)
    else:
        idmap = IndexIDMap(base_faissx_index)

    # Extract the ID mappings
    for i in range(faiss_index.ntotal):
        id_val = int(faiss_index.id_map.at(i))
        idmap._id_map[i] = id_val
        idmap._rev_id_map[id_val] = i

    # Try to extract vectors for reconstruction if possible
    try:
        if hasattr(idmap, "_vectors_by_id"):
            for i in range(faiss_index.ntotal):
                id_val = int(faiss_index.id_map.at(i))
                # Try to get the vector for caching
                try:
                    vector = faiss_index.reconstruct(i)
                    idmap._vectors_by_id[id_val] = vector
                except Exception:
                    pass  # Skip if reconstruction fails
    except Exception as e:
        logger.warning(f"Could not extract vectors from IDMap index: {e}")

    idmap.ntotal = faiss_index.ntotal
    return idmap


def _infer_index_type(faiss_index) -> Optional[str]:
    """Helper to determine the index type from a FAISS index object."""
    # Most FAISS indices have a specific class name pattern
    class_name = type(faiss_index).__name__

    if "IndexFlat" in class_name:
        return "Flat"
    elif "IndexIVFFlat" in class_name:
        nlist = getattr(faiss_index, "nlist", 100)
        return f"IVF{nlist},Flat"
    elif "IndexIVFPQ" in class_name:
        nlist = getattr(faiss_index, "nlist", 100)
        m = getattr(faiss_index.pq, "M", 8)
        nbits = getattr(faiss_index.pq, "nbits", 8)
        return f"IVF{nlist},PQ{m}x{nbits}"
    elif "IndexPQ" in class_name:
        m = getattr(faiss_index.pq, "M", 8)
        nbits = getattr(faiss_index.pq, "nbits", 8)
        return f"PQ{m}x{nbits}"
    elif "IndexHNSW" in class_name:
        m = getattr(faiss_index.hnsw, "efConstruction", 16)
        return f"HNSW{m}"
    elif "IndexScalar" in class_name:
        return "SQ8"

    return None
