from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="faiss-proxy",
    version="0.1.0",
    author="Muxi AI",
    author_email="info@muxi.ai",
    description="High-performance vector database proxy using FAISS and ZeroMQ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/muxi/faiss-proxy",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
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
        'console_scripts': [
            'faiss-proxy.server=faiss_proxy.server.cli:main',
        ],
    },
)
