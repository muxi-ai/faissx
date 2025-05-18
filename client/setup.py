from setuptools import setup, find_packages

setup(
    name="faiss_proxy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.0",
        "numpy>=1.19.0",
    ],
    description="Drop-in replacement for FAISS with remote execution capabilities",
    author="MUXI Team",
    author_email="info@muxi.com",
    url="https://github.com/muxi/faiss-proxy",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
)
