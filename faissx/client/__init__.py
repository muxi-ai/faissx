"""
FAISSx - A drop-in replacement for FAISS with remote execution capabilities via ZeroMQ
"""

import os
import sys
from typing import Optional

__version__ = "0.1.0"

# Global configuration
_API_URL: Optional[str] = os.environ.get("FAISSX_SERVER", "tcp://localhost:45678")
_API_KEY: Optional[str] = os.environ.get("FAISSX_API_KEY", "")
_TENANT_ID: Optional[str] = os.environ.get("FAISSX_TENANT_ID", "")
_FALLBACK_TO_LOCAL: bool = os.environ.get("FAISSX_FALLBACK_TO_LOCAL", "1") == "1"

# Import all public FAISS symbols
try:
    # For fallback to local FAISS when needed
    import faiss as _local_faiss
except ImportError:
    _local_faiss = None
    if _FALLBACK_TO_LOCAL:
        print("Warning: Local FAISS not found, can't use fallback mode", file=sys.stderr)
        _FALLBACK_TO_LOCAL = False

# Import client implementation
from .client import configure, FaissXClient, get_client
from .index import (
    IndexFlatL2,
    # Add other index types as they are implemented
)

# Make sure these are directly importable from the module
__all__ = [
    "configure",
    "FaissXClient",
    "get_client",
    "IndexFlatL2",
    # Add other exported symbols as they are implemented
]

# Add version for compatibility with FAISS
__version__ = _local_faiss.__version__ if _local_faiss else "1.7.0"
