#!/usr/bin/env python3
#
# High-performance vector database proxy using FAISS and ZeroMQ
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
Build script for FAISSx distributions.

The ``FAISSX_BUILD_VARIANT`` environment variable controls which distribution
metadata is emitted:

- ``cpu`` (default) -> publishes ``faissx`` with ``faiss-cpu``
- ``gpu`` -> publishes ``faissx-gpu`` with ``faiss-gpu``
"""

import importlib.util
from pathlib import Path

from setuptools import find_packages, setup

ROOT_DIR = Path(__file__).resolve().parent
BUILD_CONFIG_PATH = ROOT_DIR / "build_config.py"

build_config_spec = importlib.util.spec_from_file_location("build_config", BUILD_CONFIG_PATH)
if build_config_spec is None or build_config_spec.loader is None:
    raise RuntimeError(f"Unable to load build configuration from {BUILD_CONFIG_PATH}")

build_config = importlib.util.module_from_spec(build_config_spec)
build_config_spec.loader.exec_module(build_config)

setup(
    packages=find_packages(where=".", include=["faissx*"]),
    **build_config.get_setup_kwargs(),
)
