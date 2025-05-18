from setuptools import setup, find_packages

setup(
    name="faiss-proxy-server",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.0",
        "faiss-cpu>=1.7.0",
        "numpy>=1.20.0",
    ],
)
