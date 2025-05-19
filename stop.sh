#!/bin/bash

# Stop and remove the docker container
docker stop faiss-proxy
docker rm faiss-proxy

echo "FAISS Proxy Server stopped"
