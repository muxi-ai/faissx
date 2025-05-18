from fastapi import HTTPException, Depends, status
from fastapi.security import APIKeyHeader
import os
from typing import Dict, Optional

# Initialize API key header
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

# Simple in-memory API key to tenant ID mapping
# In production, this would be stored in a database or configuration file
API_KEYS: Dict[str, str] = {
    "test-key-1": "tenant-1",
    "test-key-2": "tenant-2",
}

# Override with environment variables if provided
def load_api_keys_from_env():
    """Load API keys from environment variables if available"""
    env_keys = os.environ.get("FAISS_PROXY_API_KEYS")
    if env_keys:
        try:
            # Format: "key1:tenant1,key2:tenant2"
            pairs = env_keys.split(",")
            for pair in pairs:
                key, tenant = pair.split(":")
                API_KEYS[key.strip()] = tenant.strip()
        except Exception as e:
            print(f"Error loading API keys from environment: {e}")


# Load API keys on startup
load_api_keys_from_env()


async def get_tenant_id(api_key: str = Depends(API_KEY_HEADER)) -> str:
    """
    Get tenant ID from API key.

    Args:
        api_key: API key from header

    Returns:
        tenant_id: Tenant ID for the API key

    Raises:
        HTTPException: If API key is invalid
    """
    if api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return API_KEYS[api_key]


def validate_tenant_access(tenant_id: str, resource_tenant_id: str) -> None:
    """
    Validate that the tenant has access to the resource.

    Args:
        tenant_id: Tenant ID from API key
        resource_tenant_id: Tenant ID of the resource

    Raises:
        HTTPException: If tenant does not have access to the resource
    """
    if tenant_id != resource_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access to this resource is forbidden",
        )
