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
Base module for common imports and utilities used by all index classes.
"""

import uuid
import numpy as np
from typing import Tuple
import faiss
import logging

# Define constants for when faiss isn't fully available
if not hasattr(faiss, 'METRIC_L2'):
    class FaissConstants:
        METRIC_L2 = 0
        METRIC_INNER_PRODUCT = 1
    faiss = FaissConstants()

from ..client import get_client

__all__ = ['uuid', 'np', 'Tuple', 'faiss', 'logging', 'get_client']
