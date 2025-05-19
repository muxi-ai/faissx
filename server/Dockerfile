FROM python:3.11-slim

LABEL maintainer="Muxi AI"
LABEL description="FAISS, Extended - A high-performance vector database proxy with ZeroMQ"
LABEL version="0.0.2"

WORKDIR /app

# Install faissx directly from PyPI
RUN pip install --no-cache-dir faissx

# Create data directory
RUN mkdir -p /data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FAISSX_DATA_DIR=/data
ENV FAISSX_PORT=45678
ENV FAISSX_BIND_ADDRESS=0.0.0.0
ENV FAISSX_ENABLE_AUTH=false

# Expose the ZeroMQ port
EXPOSE 45678

# Run the server using the CLI
# Note: For production, use docker's restart policy:
# docker run --restart=always ... or docker-compose with restart: always
CMD ["faissx.server", "run"]
