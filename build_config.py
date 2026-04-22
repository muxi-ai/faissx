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
Build-time package configuration for FAISSx distributions.

This module centralizes package metadata so the same source tree can publish
both the default CPU distribution (``faissx``) and the CUDA 12 GPU distribution
(``faissx-gpu``) without duplicating packaging files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent
README_PATH = ROOT_DIR / "README.md"
VERSION_PATH = ROOT_DIR / "faissx" / ".version"

BUILD_VARIANT_ENV_VAR = "FAISSX_BUILD_VARIANT"
DEFAULT_BUILD_VARIANT = "cpu"
VALID_BUILD_VARIANTS = {"cpu", "gpu"}

COMMON_DEPENDENCIES = [
    "numpy>=1.24.0",
    "pyzmq>=26.0.3",
    "msgpack>=1.0.8",
]

DEV_DEPENDENCIES = [
    "pytest>=8.2.2",
    "pytest-cov>=5.0.0",
    "black>=24.4.2",
    "isort>=5.13.2",
    "mypy>=1.10.0",
    "ruff>=0.4.0",
]

BASE_CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

CPU_CLASSIFIERS = BASE_CLASSIFIERS + [
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Operating System :: OS Independent",
]

GPU_CLASSIFIERS = [
    *BASE_CLASSIFIERS,
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Operating System :: POSIX :: Linux",
    "Environment :: GPU :: NVIDIA CUDA",
    "Environment :: GPU :: NVIDIA CUDA :: 12",
]


def read_version() -> str:
    """
    Read and return the package version from ``faissx/.version``.
    """
    return VERSION_PATH.read_text(encoding="utf-8").strip()


def read_long_description() -> str:
    """
    Read and return the package long description from ``README.md``.
    """
    return README_PATH.read_text(encoding="utf-8")


def get_build_variant(variant: str | None = None) -> str:
    """
    Resolve the build variant from the explicit argument or environment.

    Args:
        variant: Optional explicit variant override.

    Returns:
        The normalized build variant name.

    Raises:
        ValueError: If the selected build variant is unsupported.
    """
    selected_variant = (variant or os.getenv(BUILD_VARIANT_ENV_VAR, DEFAULT_BUILD_VARIANT)).strip()
    normalized_variant = selected_variant.lower()

    if normalized_variant not in VALID_BUILD_VARIANTS:
        valid_variants = ", ".join(sorted(VALID_BUILD_VARIANTS))
        raise ValueError(
            f"Unsupported {BUILD_VARIANT_ENV_VAR}={selected_variant!r}. "
            f"Expected one of: {valid_variants}."
        )

    return normalized_variant


def get_distribution_name(variant: str) -> str:
    """
    Return the distribution name for the requested build variant.
    """
    return "faissx-gpu" if variant == "gpu" else "faissx"


def get_install_requires(variant: str) -> list[str]:
    """
    Return install requirements for the requested build variant.
    """
    faiss_dependency = "faiss-gpu-cu12>=1.14.1.post1" if variant == "gpu" else "faiss-cpu>=1.8.0"
    return [faiss_dependency, *COMMON_DEPENDENCIES]


def get_python_requires(variant: str) -> str:
    """
    Return the supported Python range for the requested build variant.
    """
    return ">=3.10,<3.14" if variant == "gpu" else ">=3.8"


def get_classifiers(variant: str) -> list[str]:
    """
    Return PyPI classifiers for the requested build variant.
    """
    return GPU_CLASSIFIERS.copy() if variant == "gpu" else CPU_CLASSIFIERS.copy()


def get_setup_kwargs(variant: str | None = None) -> dict[str, Any]:
    """
    Return setuptools metadata for the selected build variant.
    """
    build_variant = get_build_variant(variant)

    return {
        "name": get_distribution_name(build_variant),
        "version": read_version(),
        "description": "High-performance vector database proxy using FAISS and ZeroMQ",
        "long_description": read_long_description(),
        "long_description_content_type": "text/markdown",
        "author": "Ran Aroussi",
        "author_email": "ran@aroussi.com",
        "license": "Apache-2.0",
        "url": "https://github.com/muxi-ai/faissx",
        "project_urls": {
            "Homepage": "https://github.com/muxi-ai/faissx",
            "Documentation": "https://github.com/muxi-ai/faissx#readme",
            "Issues": "https://github.com/muxi-ai/faissx/issues",
        },
        "python_requires": get_python_requires(build_variant),
        "install_requires": get_install_requires(build_variant),
        "extras_require": {
            "dev": DEV_DEPENDENCIES,
        },
        "classifiers": get_classifiers(build_variant),
        "entry_points": {
            "console_scripts": [
                "faissx.server=faissx.server.cli:main",
            ],
        },
        "include_package_data": True,
        "package_data": {
            "faissx": [".version"],
        },
        "license_files": ("LICENSE",),
    }
