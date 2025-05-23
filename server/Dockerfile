FROM python:3.11-slim

LABEL maintainer="Muxi AI"
LABEL description="FAISSx - Run FAISS as a server"

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
# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Expose the ZeroMQ port
EXPOSE 45678

# Run the server using the CLI
CMD ["faissx.server", "run"]
