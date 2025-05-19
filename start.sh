#!/bin/bash
set -e

# Create data directory if it doesn't exist
mkdir -p data

# Build and start the docker container
cd server
docker build -t faiss-proxy-zmq .
cd ..

docker run -d \
  --name faiss-proxy \
  -p 5555:5555 \
  -v "$(pwd)/data:/data" \
  -e FAISS_DATA_DIR=/data \
  -e FAISS_PROXY_PORT=5555 \
  -e PYTHONUNBUFFERED=1 \
  faiss-proxy-zmq

echo "FAISS Proxy Server started on port 5555"
echo "Use client_test.py to test the service"
