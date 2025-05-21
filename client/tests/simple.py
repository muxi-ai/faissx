#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Direct test to verify if we're truly connecting to the remote server.
This script connects to a server running on port 45678 and logs all steps.
"""

import sys
from faissx import client as faiss
from faissx.client.client import get_client
import faissx
import numpy as np

from faissx import client as faiss
import numpy as np

# Connect to a remote FAISSx server
faiss.configure(
    server="tcp://0.0.0.0:45678",  # ZeroMQ server address
    tenant_id="tenant-1"             # Tenant ID for multi-tenant isolation
)

# After configure(), all operations use the remote server
dimension = 128
index = faiss.IndexFlatL2(dimension)
vectors = np.random.random((100, dimension)).astype('float32')
index.add(vectors)
D, I = index.search(np.random.random((1, dimension)).astype('float32'), k=5)
print(D, I)
