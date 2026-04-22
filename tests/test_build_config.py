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
Tests for build-time package variant metadata.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
BUILD_CONFIG_PATH = ROOT_DIR / "build_config.py"

build_config_spec = importlib.util.spec_from_file_location("build_config", BUILD_CONFIG_PATH)
if build_config_spec is None or build_config_spec.loader is None:
    raise RuntimeError(f"Unable to load build configuration from {BUILD_CONFIG_PATH}")

build_config = importlib.util.module_from_spec(build_config_spec)
build_config_spec.loader.exec_module(build_config)

BUILD_VARIANT_ENV_VAR = build_config.BUILD_VARIANT_ENV_VAR
get_setup_kwargs = build_config.get_setup_kwargs


def test_cpu_variant_is_default(monkeypatch):
    """
    The default build should keep publishing the CPU distribution.
    """
    monkeypatch.delenv(BUILD_VARIANT_ENV_VAR, raising=False)

    setup_kwargs = get_setup_kwargs()

    assert setup_kwargs["name"] == "faissx"
    assert setup_kwargs["python_requires"] == ">=3.8"
    assert "faiss-cpu>=1.8.0" in setup_kwargs["install_requires"]
    assert "Programming Language :: Python :: 3.11" in setup_kwargs["classifiers"]


def test_gpu_variant_uses_gpu_distribution_metadata(monkeypatch):
    """
    The GPU build should emit GPU-specific distribution metadata.
    """
    monkeypatch.setenv(BUILD_VARIANT_ENV_VAR, "gpu")

    setup_kwargs = get_setup_kwargs()

    assert setup_kwargs["name"] == "faissx-gpu"
    assert setup_kwargs["python_requires"] == ">=3.10,<3.14"
    assert "faiss-gpu-cu12>=1.14.1.post1" in setup_kwargs["install_requires"]
    assert "Programming Language :: Python :: 3.13" in setup_kwargs["classifiers"]
    assert "Programming Language :: Python :: 3.9" not in setup_kwargs["classifiers"]
    assert "Operating System :: POSIX :: Linux" in setup_kwargs["classifiers"]
    assert "Environment :: GPU :: NVIDIA CUDA :: 12" in setup_kwargs["classifiers"]


def test_invalid_build_variant_raises_clear_error(monkeypatch):
    """
    Invalid build variants should fail fast with a clear error.
    """
    monkeypatch.setenv(BUILD_VARIANT_ENV_VAR, "invalid")

    with pytest.raises(ValueError, match=BUILD_VARIANT_ENV_VAR):
        get_setup_kwargs()
