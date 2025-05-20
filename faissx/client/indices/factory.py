"""
FAISSx index_factory implementation.

This module provides a FAISS-compatible index factory that parses index descriptor strings
and creates the corresponding index objects. It supports the same syntax as the original
FAISS implementation for seamless compatibility.

Example usage:
    index = index_factory(128, "Flat")  # Creates a flat index
    index = index_factory(128, "IVF100,Flat")  # Creates an IVF index with a flat quantizer
    index = index_factory(128, "IVF100,PQ16")  # Creates an IVF index with PQ
    index = index_factory(128, "HNSW32")  # Creates an HNSW index
"""

import re
import logging
from typing import Any, Optional

import faiss

from .flat import IndexFlatL2
from .ivf_flat import IndexIVFFlat
from .hnsw_flat import IndexHNSWFlat
from .pq import IndexPQ
from .ivf_pq import IndexIVFPQ
from .scalar_quantizer import IndexScalarQuantizer
from .id_map import IndexIDMap, IndexIDMap2

logger = logging.getLogger(__name__)


def index_factory(d: int, description: str, metric: Optional[int] = None) -> Any:
    """
    Parse an index description string and create the corresponding index.

    Args:
        d: Dimensionality of the vectors
        description: Index description string using FAISS syntax
        metric: Distance metric, default is L2 (faiss.METRIC_L2)

    Returns:
        A FAISSx index object corresponding to the description

    Raises:
        ValueError: If the description is malformed or unsupported
    """
    if metric is None:
        metric = faiss.METRIC_L2

    # Normalize description by removing whitespace
    description = re.sub(r'\s+', '', description)

    # Special case for IDMap and IDMap2
    if description.startswith("IDMap") or description.startswith("IDMap2"):
        is_idmap2 = description.startswith("IDMap2")
        sub_description = description[len("IDMap2,"):] if is_idmap2 else description[len("IDMap,"):]
        sub_index = index_factory(d, sub_description, metric)
        return IndexIDMap2(sub_index) if is_idmap2 else IndexIDMap(sub_index)

    # Handle Flat index (simplest case)
    if description == "Flat":
        if metric == faiss.METRIC_L2:
            return IndexFlatL2(d)
        else:
            # For now we only support L2 distance
            raise ValueError(f"Metric {metric} not supported for Flat index")

    # Handle HNSW index
    hnsw_match = re.match(r"HNSW(\d+)", description)
    if hnsw_match:
        m = int(hnsw_match.group(1))
        return IndexHNSWFlat(d, m, metric)

    # Handle IVF indices
    ivf_match = re.match(r"IVF(\d+),(\w+)", description)
    if ivf_match:
        nlist = int(ivf_match.group(1))
        coarse_quantizer_type = ivf_match.group(2)

        # Create the coarse quantizer
        if coarse_quantizer_type == "Flat":
            coarse_quantizer = IndexFlatL2(d)
            return IndexIVFFlat(coarse_quantizer, d, nlist, metric)

        # Handle IVF with Product Quantization
        pq_match = re.match(r"PQ(\d+)(x\d+)?", coarse_quantizer_type)
        if pq_match:
            m = int(pq_match.group(1))  # Number of subquantizers
            nbits = 8  # Default bits per subquantizer

            # Optional specification of bits per subquantizer
            if pq_match.group(2):
                nbits = int(pq_match.group(2)[1:])

            coarse_quantizer = IndexFlatL2(d)
            return IndexIVFPQ(coarse_quantizer, d, nlist, m, nbits, metric)

    # Handle scalar quantizer
    sq_match = re.match(r"SQ(\d+)", description)
    if sq_match:
        return IndexScalarQuantizer(d, metric)

    # Handle direct PQ index (without IVF)
    pq_match = re.match(r"PQ(\d+)(x\d+)?", description)
    if pq_match:
        m = int(pq_match.group(1))  # Number of subquantizers
        nbits = 8  # Default bits per subquantizer

        # Optional specification of bits per subquantizer
        if pq_match.group(2):
            nbits = int(pq_match.group(2)[1:])

        return IndexPQ(d, m, nbits, metric)

    # If we get here, the description was not supported
    raise ValueError(f"Unsupported index description: {description}")
