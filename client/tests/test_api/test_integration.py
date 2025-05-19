import os
import pytest
import numpy as np
import time
import subprocess
import requests
from uuid import uuid4

import faissx
from faissx import configure, IndexFlatL2


# Skip all tests in this module if NO_INTEGRATION_TESTS is set
pytestmark = pytest.mark.skipif(
    os.environ.get("NO_INTEGRATION_TESTS") == "1",
    reason="Integration tests are disabled"
)


@pytest.fixture(scope="module")
def server_process():
    """Start a FAISSx server for integration testing"""
    # Check if a server is already running
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            print("Using existing server on port 8000")
            yield None
            return
    except requests.exceptions.ConnectionError:
        pass

    print("Starting test server...")

    # Start server process
    server_cmd = ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
    process = subprocess.Popen(
        server_cmd,
        cwd=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "server"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Wait for the server to start
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/")
            if response.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            if i == max_retries - 1:
                # Last retry failed
                process.kill()
                stdout, stderr = process.communicate()
                raise RuntimeError(f"Failed to start server. Stdout: {stdout.decode()}, Stderr: {stderr.decode()}")
            time.sleep(1)

    # Server is running
    yield process

    # Teardown: kill the server process
    process.kill()
    process.wait()


@pytest.fixture(scope="module")
def client_config():
    """Configure the client for integration testing"""
    # Generate unique tenant ID for this test run
    tenant_id = f"test-tenant-{uuid4().hex[:8]}"

    # Configure the client
    configure(
        api_url="http://localhost:8000",
        api_key="integration-test-key",  # This doesn't matter for tests without auth
        tenant_id=tenant_id
    )

    return {
        "api_url": "http://localhost:8000",
        "api_key": "integration-test-key",
        "tenant_id": tenant_id
    }


class TestIntegration:
    """Integration tests for FAISSx client with a running server"""

    def test_basic_workflow(self, server_process, client_config):
        """Test basic workflow with index creation, adding vectors, and search"""
        # Create a new index
        index = IndexFlatL2(4)

        # Add vectors
        vectors = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        index.add(vectors)

        # Check count
        assert index.ntotal == 4

        # Search for similar vectors
        query = np.array([[0.9, 0.1, 0.0, 0.0]], dtype=np.float32)
        D, I = index.search(query, k=2)

        # Check results
        assert D.shape == (1, 2)
        assert I.shape == (1, 2)
        assert I[0, 0] == 0  # First vector should be the closest

        # Reset the index
        index.reset()

        # Check count after reset
        assert index.ntotal == 0

        # Try adding vectors again
        index.add(vectors[:2])
        assert index.ntotal == 2

        # Clean up
        try:
            index.__del__()
        except:
            pass

    def test_multiple_indices(self, server_process, client_config):
        """Test creating and using multiple indices"""
        # Create two indices
        index1 = IndexFlatL2(4)
        index2 = IndexFlatL2(4)

        # Add vectors to both
        vectors1 = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float32)

        vectors2 = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        index1.add(vectors1)
        index2.add(vectors2)

        # Search in first index
        query = np.array([[0.9, 0.1, 0.0, 0.0]], dtype=np.float32)
        D1, I1 = index1.search(query, k=1)

        # Search in second index
        query = np.array([[0.0, 0.0, 0.9, 0.1]], dtype=np.float32)
        D2, I2 = index2.search(query, k=1)

        # Check results
        assert I1[0, 0] == 0  # First vector in first index
        assert I2[0, 0] == 0  # First vector in second index

        # Clean up
        try:
            index1.__del__()
            index2.__del__()
        except:
            pass
