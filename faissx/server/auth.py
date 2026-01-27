#!/usr/bin/env python3
#
# Unified interface for LLM providers using OpenAI format
# https://github.com/muxi-ai/faissx
#
# Copyright (C) 2025 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FAISSx Server Authentication Module

This module handles API key authentication and tenant isolation for the FAISSx server.
It provides functions for managing API keys, loading credentials, validating requests,
enforcing tenant-level access control, and handling auth/permission failures.

The authentication system ensures multi-tenant security by restricting clients to
only access indices and vectors belonging to their assigned tenant.

Authentication Flow:
1. Client provides API key in request
2. Server validates API key against configured mappings
3. Server retrieves tenant ID associated with the API key
4. Server enforces tenant isolation for all operations

Security Model:
- Each API key maps to exactly one tenant ID
- Tenants can only access their own resources
- No cross-tenant access is permitted
- Invalid API keys are rejected immediately

Configuration Sources:
- Direct configuration via set_api_keys()
- Environment variables via load_api_keys_from_env()
- JSON files (handled by calling module)
"""

import logging
import os
from typing import Dict, Optional

# Constants for configuration and validation
ENV_API_KEYS_VAR = "faissx_API_KEYS"  # Environment variable name for API keys
ENV_KEY_SEPARATOR = ","  # Separator between key:tenant pairs in env var
ENV_PAIR_SEPARATOR = ":"  # Separator between key and tenant in each pair

# Logger for authentication-related events
logger = logging.getLogger("faissx.server")

# Global authentication state
# In-memory storage for API key to tenant ID mappings
# This provides fast O(1) lookup for authentication requests
# Structure: {"api_key_string": "tenant_id_string", ...}
# NOTE: Replace with database or config file in production for persistence
# NOTE: This is modified by set_api_keys() and load_api_keys_from_env()
API_KEYS: Dict[str, str] = {}


def set_api_keys(keys: Optional[Dict[str, str]]) -> None:
    """
    Initialize the global API_KEYS dictionary with provided key-tenant mappings.

    This function replaces the entire authentication configuration with the provided
    mappings. It's designed to be called during server initialization to establish
    the complete set of valid API keys and their associated tenants.

    Thread Safety:
        This function modifies global state and should only be called during
        server initialization before handling any requests.

    Args:
        keys: Dictionary mapping API keys to their corresponding tenant IDs.
              Can be None or empty to disable authentication.
              Keys and values must be non-empty strings.

    Note:
        Creates a defensive copy to prevent external modifications to the internal
        authentication state. The original dictionary can be safely modified
        after this call without affecting server authentication.

    Example:
        set_api_keys({
            "prod-key-123": "tenant-production",
            "dev-key-456": "tenant-development"
        })
    """
    global API_KEYS

    # Handle None input gracefully - empty dict means no authentication
    # This allows callers to explicitly disable auth by passing None
    if keys is None:
        API_KEYS = {}
        logger.info("Authentication disabled - no API keys configured")
        return

    # Create defensive copy to prevent external state modification
    # This ensures the authentication state remains stable during server operation
    # Even if the caller modifies their original dictionary after this call
    API_KEYS = keys.copy()

    # Log configuration summary for debugging (without exposing actual keys)
    # We count unique tenants to help administrators understand the setup
    key_count = len(API_KEYS)
    tenant_count = len(set(API_KEYS.values()))  # Count unique tenant IDs
    logger.info(
        f"Authentication configured with {key_count} API keys for {tenant_count} tenants"
    )


def load_api_keys_from_env() -> None:
    """
    Load API key configurations from environment variables.

    Reads the environment variable specified by ENV_API_KEYS_VAR and parses it
    into API key to tenant ID mappings. The format should be comma-separated
    key:tenant pairs.

    Environment Variable Format:
        faissx_API_KEYS="key1:tenant1,key2:tenant2,key3:tenant3"

    Parsing Behavior:
        - Whitespace around keys and tenant IDs is automatically stripped
        - Empty pairs are skipped silently
        - Malformed pairs cause the entire loading operation to fail
        - Duplicate keys will overwrite previous values (last one wins)

    Error Handling:
        Parsing errors are logged as critical and do not raise exceptions,
        allowing the server to continue with existing authentication configuration.

    Note:
        This function modifies the global API_KEYS dictionary. It should be called
        during server initialization, preferably before handling any requests.

    Raises:
        No exceptions are raised - all errors are logged and handled gracefully.
    """
    global API_KEYS

    # Attempt to read the environment variable
    # Environment variables are commonly used in containerized deployments
    env_keys = os.environ.get(ENV_API_KEYS_VAR)
    if not env_keys:
        logger.debug(f"No API keys found in environment variable {ENV_API_KEYS_VAR}")
        return

    logger.info(f"Loading API keys from environment variable {ENV_API_KEYS_VAR}")

    try:
        # Split into individual key:tenant pairs using the configured separator
        # Example: "key1:tenant1,key2:tenant2" becomes ["key1:tenant1", "key2:tenant2"]
        pairs = env_keys.split(ENV_KEY_SEPARATOR)
        loaded_count = 0

        # Process each key:tenant pair individually
        for pair in pairs:
            # Skip empty pairs that might result from trailing commas or extra spaces
            # This makes the parsing more forgiving of common formatting mistakes
            pair = pair.strip()
            if not pair:
                continue

            # Validate the expected key:tenant format before attempting to parse
            # This provides better error messages than a generic split failure
            if ENV_PAIR_SEPARATOR not in pair:
                raise ValueError(
                    f"Invalid pair format '{pair}' - expected 'key:tenant'"
                )

            # Split on first colon only, in case tenant IDs contain colons
            # maxsplit=1 ensures we only split on the first occurrence
            key, tenant = pair.split(ENV_PAIR_SEPARATOR, 1)
            key = key.strip()  # Remove any surrounding whitespace
            tenant = tenant.strip()  # Remove any surrounding whitespace

            # Validate that both key and tenant are non-empty after stripping
            # Empty values would create security vulnerabilities or lookup failures
            if not key or not tenant:
                raise ValueError(f"Empty key or tenant in pair '{pair}'")

            # Store the validated mapping in the global authentication dictionary
            # This overwrites any existing mapping for the same key (last one wins)
            API_KEYS[key] = tenant
            loaded_count += 1

        logger.info(
            f"Successfully loaded {loaded_count} API key mappings from environment"
        )

    except Exception as e:
        # Log critical error but don't crash the server - graceful degradation
        # This allows the server to start with existing auth config or no auth
        logger.critical(
            f"Error loading API keys from environment variable {ENV_API_KEYS_VAR}: {e}"
        )
        logger.critical(
            "Server will continue with existing authentication configuration"
        )


def get_tenant_id(api_key: str) -> Optional[str]:
    """
    Retrieve the tenant ID associated with a given API key.

    Performs a fast O(1) lookup in the configured API key mappings to find the
    tenant ID associated with the provided API key. This is the core function
    used during authentication to map client credentials to tenant identities.

    Args:
        api_key: The API key to look up from request headers or authentication data.
                Must be a non-None string (empty strings will return None).

    Returns:
        str: The tenant ID associated with the API key if found
        None: If the API key is not found in the configured mappings

    Security Note:
        This function does not log the API key value to prevent credential leakage
        in log files. Only the lookup result (success/failure) should be logged
        by calling code.

    Example:
        tenant = get_tenant_id("my-api-key-123")
        if tenant:
            print(f"Authenticated as tenant: {tenant}")
        else:
            print("Invalid API key")
    """
    # Perform O(1) lookup in the API key dictionary
    # Returns None automatically if key is not found (dict.get() behavior)
    # This is safe even if API_KEYS is empty or api_key is None/empty
    return API_KEYS.get(api_key)


def validate_tenant_access(tenant_id: str, resource_tenant_id: str) -> bool:
    """
    Verify if a tenant has permission to access a specific resource.

    Implements the core authorization logic for multi-tenant isolation. Currently
    uses a simple ownership model where tenants can only access resources that
    belong to their own tenant. This ensures strict isolation between tenants.

    Security Model:
        - Strict ownership: tenants can only access their own resources
        - No cross-tenant access is permitted
        - No inheritance or delegation of permissions
        - Case-sensitive tenant ID comparison

    Future Extensions:
        This function could be extended to support:
        - Role-based access control (RBAC)
        - Resource sharing between specific tenants
        - Administrative super-user privileges
        - Time-based access permissions

    Args:
        tenant_id: The tenant ID extracted from the authenticated API key.
                  Must be a non-empty string representing the requesting tenant.
        resource_tenant_id: The tenant ID that owns the resource being accessed.
                           Must be a non-empty string representing the resource owner.

    Returns:
        bool: True if access is allowed (tenant IDs match), False if denied

    Example:
        # Same tenant - access granted
        assert validate_tenant_access("tenant-a", "tenant-a") == True

        # Different tenants - access denied
        assert validate_tenant_access("tenant-a", "tenant-b") == False
    """
    # Simple ownership check: tenant can only access their own resources
    # This provides strict multi-tenant isolation through case-sensitive comparison
    # Returns False for any mismatch, including case differences
    return tenant_id == resource_tenant_id


class AuthError(Exception):
    """
    Exception raised when API key validation fails.

    This exception indicates that the authentication step has failed, meaning the
    provided API key is either invalid, missing, or malformed. It should be caught
    by request handlers to return appropriate HTTP 401 Unauthorized responses.

    Common Scenarios:
        - API key not found in configured mappings
        - Empty or None API key provided
        - Malformed API key format

    Security Note:
        Error messages should not reveal whether a specific API key exists to
        prevent enumeration attacks. Use generic "Invalid API key" messages.

    Example:
        try:
            authenticate_request(api_key)
        except AuthError:
            return {"error": "Invalid API key"}, 401
    """

    def __init__(self, message: str = "Authentication failed"):
        """
        Initialize AuthError with descriptive message.

        This constructor allows customization of the error message while providing
        a sensible default. The message is stored both in the exception (via super())
        and as an instance attribute for easy access by error handlers.

        Args:
            message: Human-readable error description for logging/debugging.
                    Should not contain sensitive information like API keys.

        Note:
            The message will be available both via str(exception) and
            exception.message for flexible error handling patterns.
        """
        super().__init__(message)  # Set the exception message
        self.message = message  # Store for programmatic access


class PermissionError(Exception):
    """
    Exception raised when a tenant attempts unauthorized resource access.

    This exception indicates that while authentication succeeded (valid API key),
    the authorization check failed. The tenant lacks permission to access the
    requested resource, typically because it belongs to a different tenant.

    Common Scenarios:
        - Cross-tenant resource access attempts
        - Accessing resources without proper ownership
        - Attempting operations outside tenant scope

    Security Implications:
        This exception helps enforce multi-tenant isolation by preventing data
        leakage between tenants. All resource access should be validated through
        the authorization system.

    Example:
        try:
            authenticate_request(api_key, resource_tenant_id)
        except PermissionError:
            return {"error": "Access denied"}, 403
    """

    def __init__(self, message: str = "Permission denied"):
        """
        Initialize PermissionError with descriptive message.

        This constructor allows customization of the error message while providing
        a sensible default. The message is stored both in the exception (via super())
        and as an instance attribute for easy access by error handlers.

        Args:
            message: Human-readable error description for logging/debugging.
                    May contain tenant IDs for debugging but should not expose
                    sensitive resource details to unauthorized clients.

        Note:
            The message will be available both via str(exception) and
            exception.message for flexible error handling patterns.
        """
        super().__init__(message)  # Set the exception message
        self.message = message  # Store for programmatic access


def authenticate_request(api_key: str, resource_tenant_id: Optional[str] = None) -> str:
    """
    Perform complete authentication and authorization for an incoming request.

    This is the main entry point for validating client requests. It implements a
    two-step security process: first authenticating the API key to establish client
    identity, then authorizing access to specific resources if requested.

    Authentication Flow:
        1. Look up the API key in configured mappings
        2. Extract the tenant ID associated with the key
        3. If resource access is requested, verify tenant ownership
        4. Return the authenticated tenant ID for request processing

    Security Features:
        - Fast O(1) API key lookup for performance
        - Strict tenant isolation enforcement
        - Generic error messages to prevent information disclosure
        - Comprehensive logging for security monitoring

    Args:
        api_key: The API key extracted from the request headers or body.
                Must be a non-empty string provided by the client.
        resource_tenant_id: Optional tenant ID of the specific resource being
                           accessed. If provided, authorization check will be performed
                           to ensure the authenticated tenant has permission to access
                           this resource.

    Returns:
        str: The tenant ID associated with the authenticated API key.
             This can be used by the calling code to scope all operations
             to the appropriate tenant context.

    Raises:
        AuthError: When the API key is invalid, missing, or not found in the
                  configured mappings. This indicates authentication failure.
        PermissionError: When the API key is valid but the associated tenant
                        does not have permission to access the requested resource.
                        This indicates authorization failure.

    Example:
        # Simple authentication
        tenant_id = authenticate_request("my-api-key")

        # Authentication with resource access check
        tenant_id = authenticate_request("my-api-key", "target-tenant-id")

        # Error handling
        try:
            tenant_id = authenticate_request(request.api_key)
            # Process request for tenant_id
        except AuthError:
            return error_response("Invalid API key", 401)
        except PermissionError:
            return error_response("Access denied", 403)
    """
    # Step 1: Authentication - validate API key and extract tenant identity
    # This performs an O(1) lookup in the configured API key mappings
    # The lookup returns None if the key is not found or if authentication is disabled
    tenant_id = get_tenant_id(api_key)
    if tenant_id is None:
        # Use generic error message to prevent API key enumeration attacks
        # Don't reveal whether the key exists or not - just say it's invalid
        raise AuthError("Invalid API key")

    # Step 2: Authorization - verify resource access permissions if required
    # This step is only performed when specific resource access is requested
    # If resource_tenant_id is None, we skip authorization (global access)
    if resource_tenant_id is not None:
        # Check if the authenticated tenant owns the requested resource
        # This implements strict tenant isolation - no cross-tenant access
        if not validate_tenant_access(tenant_id, resource_tenant_id):
            # Log the authorization failure for security monitoring and auditing
            # Include both tenant IDs for forensic analysis without exposing API keys
            logger.warning(
                f"Authorization denied: tenant '{tenant_id}' attempted to access "
                f"resource owned by tenant '{resource_tenant_id}'"
            )
            # Raise permission error with details for debugging (logs only)
            # The calling code should return a generic "Access denied" to the client
            raise PermissionError(
                f"Tenant {tenant_id} does not have access to resource owned by {resource_tenant_id}"
            )

    # Step 3: Return the authenticated tenant ID for request processing
    # The calling code can use this to scope all operations to the correct tenant
    # This tenant ID should be used as a filter for all database queries and operations
    return tenant_id
