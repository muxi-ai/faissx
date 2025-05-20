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
from typing import Any, Optional

import numpy as np
import faiss

from .flat import IndexFlatL2
from .ivf_flat import IndexIVFFlat
from .hnsw_flat import IndexHNSWFlat
from .pq import IndexPQ
from .ivf_pq import IndexIVFPQ
from .scalar_quantizer import IndexScalarQuantizer
from .id_map import IndexIDMap, IndexIDMap2
from .factory import index_factory

logger = logging.getLogger(__name__)


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
    dirname = os.path.dirname(fname)
    if dirname and not os.path.isdir(dirname):
        os.makedirs(dirname, exist_ok=True)

    # Handle local mode indices directly using FAISS's persistence
    if hasattr(index, "_local_index") and index._local_index is not None:
        try:
            logger.info(f"Saving index to {fname} using local FAISS implementation")
            faiss.write_index(index._local_index, fname)
            return
        except Exception as e:
            logger.error(f"Error saving local index: {e}")
            raise ValueError(f"Failed to save index: {e}")

    # Special handling for IDMap and IDMap2 indices to preserve ID mappings
    if isinstance(index, (IndexIDMap, IndexIDMap2)):
        # Use temporary file for intermediate storage
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            temp_path = tmp.name
            try:
                # Convert ID mappings to numpy array for efficient storage
                id_map = np.array(
                    [
                        (internal_idx, ext_id)
                        for internal_idx, ext_id in index._id_map.items()
                    ],
                    dtype=[("internal_idx", np.int64), ("external_id", np.int64)],
                )

                # Reconstruct vectors for storage
                vectors = []
                for i in range(index.ntotal):
                    if i in index._id_map:  # Skip removed vectors
                        vectors.append(index.reconstruct(index._id_map[i]))

                vectors = (
                    np.vstack(vectors)
                    if vectors
                    else np.zeros((0, index.d), dtype=np.float32)
                )

                # Save base index to temporary file
                faiss.write_index(
                    (
                        index.index._local_index
                        if hasattr(index.index, "_local_index")
                        else index.index
                    ),
                    temp_path,
                )

                # Read index data for combined storage
                with open(temp_path, "rb") as f:
                    index_data = f.read()

                # Write combined file with format:
                # [is_idmap2: byte][ntotal: int64][dim: int64]
                # [id_map_size: int64][id_map: bytes]
                # [index_data_size: int64][index_data: bytes]
                with open(fname, "wb") as f:
                    # Write IDMap type flag
                    f.write(bytes([1 if isinstance(index, IndexIDMap2) else 0]))

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

                log_idx = '2' if isinstance(index, IndexIDMap2) else ''
                logger.info(f"Successfully saved IndexIDMap{log_idx} to {fname}")

            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        return

    # Handle other remote indices by reconstructing in memory
    try:
        logger.info(f"Saving index to {fname} using reconstruction approach")

        # Validate index state
        if not getattr(index, "is_trained", True):
            raise ValueError("Cannot save untrained index")

        if getattr(index, "ntotal", 0) == 0:
            # Create empty index of same type for empty indices
            index_type = type(index).__name__
            dummy_index = None

            # Create appropriate empty index based on type
            if isinstance(index, IndexFlatL2):
                dummy_index = faiss.IndexFlatL2(index.d)
            elif isinstance(index, IndexIVFFlat):
                quantizer = faiss.IndexFlatL2(index.d)
                dummy_index = faiss.IndexIVFFlat(quantizer, index.d, index.nlist)
            elif isinstance(index, IndexPQ):
                dummy_index = faiss.IndexPQ(index.d, index.m, index.nbits)
            elif isinstance(index, IndexIVFPQ):
                quantizer = faiss.IndexFlatL2(index.d)
                dummy_index = faiss.IndexIVFPQ(
                    quantizer, index.d, index.nlist, index.m, index.nbits
                )
            elif isinstance(index, IndexHNSWFlat):
                dummy_index = faiss.IndexHNSWFlat(index.d, index.m)
            elif isinstance(index, IndexScalarQuantizer):
                dummy_index = faiss.IndexScalarQuantizer(index.d)
            else:
                raise ValueError(f"Unsupported index type for saving: {index_type}")

            faiss.write_index(dummy_index, fname)
            return

        # Reconstruct all vectors
        vectors = []
        for i in range(index.ntotal):
            try:
                vectors.append(index.reconstruct(i))
            except Exception:
                # Skip vectors that can't be reconstructed
                continue

        if not vectors:
            raise ValueError("No vectors could be reconstructed for saving")

        vectors = np.vstack(vectors)

        # Create new index of appropriate type
        index_type = type(index).__name__
        local_index = None

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
            raise ValueError(f"Unsupported index type for saving: {index_type}")

        # Train index if needed
        if not getattr(local_index, "is_trained", True):
            local_index.train(vectors)

        # Add vectors and save
        local_index.add(vectors)
        faiss.write_index(local_index, fname)

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
    # Import faiss here to avoid circular imports
    import faiss

    if not os.path.isfile(fname):
        raise ValueError(f"File not found: {fname}")

    # Try reading as custom IDMap format first
    try:
        with open(fname, "rb") as f:
            # Check if file is in custom IDMap format
            is_idmap_file = f.read(1)[0]

            if is_idmap_file in [0, 1]:  # 0 = IDMap, 1 = IDMap2
                is_idmap2 = is_idmap_file == 1

                # Read index dimensions
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

                # Read vectors (skip processing but read the data from the file)
                vectors_size = np.frombuffer(f.read(8), dtype=np.int64)[0]
                f.read(vectors_size)  # Read but don't process the vectors bytes

                # Create a temporary file for the index
                with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as tmp:
                    temp_path = tmp.name
                    try:
                        tmp.write(index_data)
                        tmp.flush()

                        # Read the base index
                        base_index = faiss.read_index(temp_path)

                        # If GPU is requested, move to GPU
                        if gpu:
                            try:
                                import faiss.contrib.gpu  # type: ignore

                                if faiss.get_num_gpus() > 0:
                                    res = faiss.StandardGpuResources()
                                    base_index = faiss.index_cpu_to_gpu(
                                        res, 0, base_index
                                    )
                            except (ImportError, AttributeError):
                                logger.warning(
                                    "GPU requested but FAISS GPU support not available"
                                )

                        # Create the corresponding FAISSx index
                        if isinstance(base_index, faiss.IndexFlatL2):
                            faissx_base = IndexFlatL2(d)
                            faissx_base._local_index = base_index
                        elif isinstance(base_index, faiss.IndexIVFFlat):
                            # Need to wrap the quantizer first
                            quant = IndexFlatL2(d)
                            quant._local_index = base_index.quantizer
                            faissx_base = IndexIVFFlat(quant, d, base_index.nlist)
                            faissx_base._local_index = base_index
                            faissx_base._nprobe = base_index.nprobe
                        elif isinstance(base_index, faiss.IndexPQ):
                            faissx_base = IndexPQ(
                                d, base_index.pq.M, base_index.pq.nbits
                            )
                            faissx_base._local_index = base_index
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
                        elif isinstance(base_index, faiss.IndexHNSWFlat):
                            faissx_base = IndexHNSWFlat(
                                d, base_index.hnsw.efConstruction
                            )
                            faissx_base._local_index = base_index
                        elif isinstance(base_index, faiss.IndexScalarQuantizer):
                            faissx_base = IndexScalarQuantizer(d)
                            faissx_base._local_index = base_index
                        else:
                            raise ValueError(
                                f"Unsupported base index type: {type(base_index)}"
                            )

                        # Create the IDMap/IDMap2 wrapper
                        wrapper_class = IndexIDMap2 if is_idmap2 else IndexIDMap
                        idmap_index = wrapper_class(faissx_base)

                        # Restore the internal state
                        idmap_index.ntotal = ntotal
                        idmap_index._id_map = {row[0]: row[1] for row in id_map_data}
                        idmap_index._rev_id_map = {
                            row[1]: row[0] for row in id_map_data
                        }

                        logger.info(
                            f"Successfully loaded IndexIDMap{'2' if is_idmap2 else ''} from {fname}"
                        )
                        return idmap_index

                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)

    except (IOError, ValueError, IndexError) as e:
        # Not a custom IDMap file or error reading it, try standard FAISS reading
        logger.debug(f"Not a custom IDMap file, trying standard FAISS reading: {e}")
        pass

    # Standard FAISS index reading
    try:
        logger.info(f"Loading index from {fname} using standard FAISS")
        faiss_index = faiss.read_index(fname)

        # If GPU is requested, move to GPU
        if gpu:
            try:
                import faiss.contrib.gpu  # type: ignore

                if faiss.get_num_gpus() > 0:
                    res = faiss.StandardGpuResources()
                    faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
            except (ImportError, AttributeError):
                logger.warning("GPU requested but FAISS GPU support not available")

        # Create the corresponding FAISSx index
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
            index.is_trained = faiss_index.is_trained
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
            # It's an IDMap, need to wrap the base index
            base_index = read_index(fname + ".tmp", gpu=gpu)

            if hasattr(faiss_index, "replace_vector"):
                idmap = IndexIDMap2(base_index)
            else:
                idmap = IndexIDMap(base_index)

            # Try to extract the ID mappings
            for i in range(faiss_index.ntotal):
                idmap._id_map[i] = faiss_index.id_map.at(i)
                idmap._rev_id_map[faiss_index.id_map.at(i)] = i

            idmap.ntotal = faiss_index.ntotal
            return idmap

        else:
            # Let's fall back to a best-guess approach
            index_description = _infer_index_type(faiss_index)
            if index_description:
                logger.info(f"Automatically detected index type: {index_description}")
                return index_factory(d, index_description)
            else:
                raise ValueError(f"Unsupported index type: {type(faiss_index)}")

    except Exception as e:
        logger.error(f"Error loading index: {e}")
        raise ValueError(f"Failed to load index: {e}")


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
