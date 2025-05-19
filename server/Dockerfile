FROM python:3.10-slim

LABEL maintainer="Muxi AI"
LABEL description="FAISSx Server - High-performance vector database proxy with ZeroMQ"
LABEL version="0.1.0"

WORKDIR /app

# Copy the package files and install faissx
COPY setup.py README.md MANIFEST.in requirements.txt ./
COPY faissx ./faissx
COPY examples ./examples
RUN pip install --no-cache-dir -e .

# Copy the run_server script
COPY server/run_server.py .

# Create data directory
RUN mkdir -p /data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FAISS_DATA_DIR=/data
ENV FAISS_PORT=45678
ENV FAISS_BIND_ADDRESS=0.0.0.0
ENV FAISS_ENABLE_AUTH=false

# Expose the ZeroMQ port
EXPOSE 45678

# Run the server
CMD ["python", "run_server.py"]
