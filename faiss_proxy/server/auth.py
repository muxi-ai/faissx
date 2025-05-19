import os
from typing import Dict, Optional

# Simple in-memory API key to tenant ID mapping
# In production, this would be stored in a database or configuration file
API_KEYS: Dict[str, str] = {}


# Function to set API keys programmatically (used by server.configure)
def set_api_keys(keys: Dict[str, str]):
    """Set API keys programmatically"""
    global API_KEYS
    API_KEYS = keys.copy() if keys else {}


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


def get_tenant_id(api_key: str) -> Optional[str]:
    """
    Get tenant ID from API key.

    Args:
        api_key: API key from header

    Returns:
        tenant_id: Tenant ID for the API key or None if invalid
    """
    return API_KEYS.get(api_key)


def validate_tenant_access(tenant_id: str, resource_tenant_id: str) -> bool:
    """
    Validate that the tenant has access to the resource.

    Args:
        tenant_id: Tenant ID from API key
        resource_tenant_id: Tenant ID of the resource

    Returns:
        bool: True if tenant has access, False otherwise
    """
    return tenant_id == resource_tenant_id


class AuthError(Exception):
    """Authentication error"""

    pass


class PermissionError(Exception):
    """Permission error"""

    pass


def authenticate_request(api_key: str, resource_tenant_id: Optional[str] = None):
    """
    Authenticate a request and validate tenant access if applicable.

    Args:
        api_key: API key from request
        resource_tenant_id: Tenant ID of the resource (if applicable)

    Returns:
        str: Tenant ID

    Raises:
        AuthError: If API key is invalid
        PermissionError: If tenant does not have access to the resource
    """
    tenant_id = get_tenant_id(api_key)
    if tenant_id is None:
        raise AuthError("Invalid API key")

    if resource_tenant_id is not None and not validate_tenant_access(
        tenant_id, resource_tenant_id
    ):
        raise PermissionError(
            f"Tenant {tenant_id} does not have access to resource owned by {resource_tenant_id}"
        )

    return tenant_id
