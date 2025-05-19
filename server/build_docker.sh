#!/bin/bash
# Build the Docker image for FAISS Proxy Server

set -e

# Go to project root directory (one level up from server/)
cd "$(dirname "$0")/.."

echo "Building FAISS Proxy Server Docker image..."
docker build -t muxi/faiss-proxy:latest -f server/Dockerfile .

echo "Build complete!"
echo "To run the server:"
echo "docker run -p 45678:45678 -v \$(pwd)/data:/data muxi/faiss-proxy:latest"
echo ""
echo "Or with docker-compose:"
echo "docker-compose up -d"
