"""
FAISS Proxy - A drop-in replacement for FAISS with remote execution capabilities
"""

import os
import sys
from typing import Optional

__version__ = "0.1.0"

# Global configuration
_API_URL: Optional[str] = os.environ.get("FAISS_PROXY_API_URL")
_API_KEY: Optional[str] = os.environ.get("FAISS_PROXY_API_KEY")
_TENANT_ID: Optional[str] = os.environ.get("FAISS_PROXY_TENANT_ID")
_FALLBACK_TO_LOCAL: bool = os.environ.get("FAISS_PROXY_FALLBACK_TO_LOCAL", "1") == "1"

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
from .client import configure
from .index import (
    IndexFlatL2,
    # Add other index types as they are implemented
)

# Make sure these are directly importable from the module
__all__ = [
    "configure",
    "IndexFlatL2",
    # Add other exported symbols as they are implemented
]

# Add version for compatibility with FAISS
__version__ = _local_faiss.__version__ if _local_faiss else "1.7.0"
