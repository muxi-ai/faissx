import os
from setuptools import setup, find_packages

# Read version from .version file in the faissx package
version_file = os.path.join(os.path.dirname(__file__), "faissx", ".version")
try:
    with open(version_file, "r") as f:
        version = f.read().strip()
except (IOError, FileNotFoundError):
    print(f"Warning: Could not read version from {version_file}, using default version")
    version = "0.1.0"  # Default version if .version file is missing

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="FAISSx",
    version=version,
    author="Ran Aroussi",
    author_email="ran@aroussi.com",
    description="High-performance vector database proxy using FAISS and ZeroMQ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/muxi-ai/faissx",
    packages=find_packages(),
    include_package_data=True,  # Include non-Python files specified in MANIFEST.in
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",  # Updated to match Apache 2.0
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "faiss-cpu>=1.7.2",  # or faiss-gpu for GPU support
        "numpy>=1.19.5",
        "pyzmq>=22.0.0",
        "msgpack>=1.0.2",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "isort>=5.9.1",
            "mypy>=0.812",
        ],
    },
    entry_points={
        "console_scripts": [
            "faissx.server=faissx.server.cli:main",
        ],
    },
)
