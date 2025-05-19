import os
import pytest
import tempfile
import zmq
import threading
import time
import atexit
import sys

# Add the server directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.faiss_core import get_faiss_manager
import src.auth as auth
from src.run import FaissProxyServer


# Global server instance for testing
_test_server = None
_test_server_thread = None


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data"""
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory for tests: {temp_dir}")
        # Make sure the directory exists
        os.makedirs(temp_dir, exist_ok=True)
        yield temp_dir


@pytest.fixture(scope="function", autouse=True)
def reset_faiss_manager_singleton(test_data_dir):
    """Reset the FaissManager singleton for each test to use test_data_dir"""
    global _faiss_manager_instance

    # Set the environment variable for the data dir
    os.environ["FAISS_DATA_DIR"] = test_data_dir
    print(f"Set FAISS_DATA_DIR to: {test_data_dir}")

    # Reset the singleton
    _faiss_manager_instance = None

    # Get a fresh instance for the test
    manager = get_faiss_manager()
    print(f"Reset FaissManager singleton with data_dir: {manager.data_dir}")

    yield manager


@pytest.fixture(scope="function")
def test_tenant():
    """Get a test tenant ID"""
    return "test-tenant-001"


@pytest.fixture(scope="function")
def test_api_key():
    """Get a test API key"""
    return "test-api-key-001"


@pytest.fixture(scope="function", autouse=True)
def setup_test_auth(monkeypatch, test_tenant, test_api_key):
    """Setup authentication for testing"""
    # Add our test API key to the API_KEYS dictionary
    auth.API_KEYS[test_api_key] = test_tenant
    print(f"Added test API key {test_api_key} for tenant {test_tenant}")

    yield

    # Clean up
    if test_api_key in auth.API_KEYS:
        del auth.API_KEYS[test_api_key]


def start_test_server():
    """Start a test server in a background thread"""
    global _test_server, _test_server_thread

    if _test_server_thread and _test_server_thread.is_alive():
        return

    # Use an ephemeral port for testing
    _test_server = FaissProxyServer(bind_address="127.0.0.1", port=0)

    def run_server():
        _test_server.start()

    _test_server_thread = threading.Thread(target=run_server, daemon=True)
    _test_server_thread.start()

    # Wait for server to start and get the assigned port
    time.sleep(0.5)

    # Register cleanup
    atexit.register(stop_test_server)

    return _test_server


def stop_test_server():
    """Stop the test server"""
    global _test_server, _test_server_thread

    if _test_server:
        _test_server.stop()
        if _test_server_thread:
            _test_server_thread.join(timeout=2)
        _test_server = None
        _test_server_thread = None


@pytest.fixture(scope="module")
def zmq_server():
    """Start a ZeroMQ server for testing"""
    server = start_test_server()
    yield server
    stop_test_server()


@pytest.fixture(scope="function")
def zmq_client(zmq_server):
    """Create a ZeroMQ client for testing"""
    context = zmq.Context()
    socket = context.socket(zmq.REQ)

    # Connect to the running server
    bind_url = f"tcp://127.0.0.1:{zmq_server.port}"
    socket.connect(bind_url)

    yield socket

    socket.close()
    context.term()
