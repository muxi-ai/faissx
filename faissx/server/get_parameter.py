#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FAISSx Server Get Parameter
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
Get parameter support for FAISSx Server

This module provides enhanced parameter retrieval capabilities for
various FAISS index types, including specialized ones like PQ, SQ,
and binary indices.
"""

import faiss
import logging

logger = logging.getLogger(__name__)

def get_parameter(self, index_id, param_name):
    """
    Get the current value of a parameter for the specified index.

    Args:
        index_id (str): ID of the index
        param_name (str): Name of the parameter to retrieve

    Returns:
        dict: Response containing the parameter value or error message
    """
    if index_id not in self.indexes:
        return {"success": False, "error": f"Index {index_id} not found"}

    index = self.indexes[index_id]

    try:
        # IVF index parameters
        if isinstance(index, faiss.IndexIVF):
            if param_name == "nprobe":
                return {
                    "success": True,
                    "param_name": "nprobe",
                    "param_value": index.nprobe
                }
            elif param_name == "nlist":
                return {
                    "success": True,
                    "param_name": "nlist",
                    "param_value": index.nlist
                }
            elif param_name == "quantizer_type" and hasattr(index, "quantizer"):
                return {
                    "success": True,
                    "param_name": "quantizer_type",
                    "param_value": type(index.quantizer).__name__
                }

        # HNSW index parameters
        elif isinstance(index, faiss.IndexHNSW):
            if param_name == "efSearch":
                return {
                    "success": True,
                    "param_name": "efSearch",
                    "param_value": index.hnsw.efSearch
                }
            elif param_name == "efConstruction":
                return {
                    "success": True,
                    "param_name": "efConstruction",
                    "param_value": index.hnsw.efConstruction
                }
            elif param_name == "M":
                return {
                    "success": True,
                    "param_name": "M",
                    "param_value": index.hnsw.M
                }

        # Product Quantization (PQ) index parameters
        elif isinstance(index, faiss.IndexPQ):
            if param_name == "M":
                return {
                    "success": True,
                    "param_name": "M",
                    "param_value": index.pq.M
                }
            elif param_name == "nbits":
                return {
                    "success": True,
                    "param_name": "nbits",
                    "param_value": index.pq.nbits
                }
            elif param_name == "use_precomputed_table":
                return {
                    "success": True,
                    "param_name": "use_precomputed_table",
                    "param_value": getattr(index, "use_precomputed_table", False)
                }

        # Scalar Quantization (SQ) index parameters
        elif isinstance(index, faiss.IndexScalarQuantizer):
            if param_name == "qtype":
                return {
                    "success": True,
                    "param_name": "qtype",
                    "param_value": str(index.sq_type)
                }

        # Binary index parameters
        elif isinstance(index, faiss.IndexBinaryIVF):
            if param_name == "nprobe":
                return {
                    "success": True,
                    "param_name": "nprobe",
                    "param_value": index.nprobe
                }
            elif param_name == "nlist":
                return {
                    "success": True,
                    "param_name": "nlist",
                    "param_value": index.nlist
                }

        # Binary hash index parameters
        elif isinstance(index, faiss.IndexBinaryHash):
            if param_name == "bits_per_dim":
                return {
                    "success": True,
                    "param_name": "bits_per_dim",
                    "param_value": index.b
                }

        # Parameters for all index types
        if param_name == "is_trained":
            # This parameter is available for all index types
            is_trained = getattr(index, "is_trained", True)
            return {
                "success": True,
                "param_name": "is_trained",
                "param_value": is_trained
            }
        elif param_name == "dimension":
            return {
                "success": True,
                "param_name": "dimension",
                "param_value": self.dimensions[index_id]
            }
        elif param_name == "ntotal":
            return {
                "success": True,
                "param_name": "ntotal",
                "param_value": index.ntotal
            }
        elif param_name == "metric_type":
            if hasattr(index, "metric_type"):
                metric_type = index.metric_type
                metric_name = "L2"
                if metric_type == faiss.METRIC_INNER_PRODUCT:
                    metric_name = "IP"
                return {
                    "success": True,
                    "param_name": "metric_type",
                    "param_value": metric_name
                }

        # If we get here, the parameter is not supported
        return {
            "success": False,
            "error": f"Parameter {param_name} not supported for this index type ({type(index).__name__})"
        }

    except Exception as e:
        logger.exception(f"Error getting parameter: {e}")
        return {"success": False, "error": f"Error getting parameter: {str(e)}"}
