#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

This module handles API key authentication and tenant isolation for
the FAISSx server.

It provides functions for:
- Managing API keys and their associated tenant IDs
- Loading auth credentials from environment variables or configuration
- Validating requests based on API keys
- Enforcing tenant-level access control to ensure data isolation
- Custom exceptions for authentication and permission failures

The authentication system ensures that each client can only access indices and
vectors that belong to their assigned tenant, providing multi-tenant security.
"""

import os
from typing import Dict, Optional

# Simple in-memory API key to tenant ID mapping
# In production, this would be stored in a database or configuration file
API_KEYS: Dict[str, str] = {}


def set_api_keys(keys: Dict[str, str]):
    """
    Set API keys programmatically in the global API_KEYS dictionary.

    This function is used by server.configure to initialize API keys.
    It creates a copy of the input dictionary to prevent external modifications.

    Args:
        keys: Dictionary mapping API keys to tenant IDs
    """
    global API_KEYS
    API_KEYS = keys.copy() if keys else {}


def load_api_keys_from_env():
    """
    Load API keys from environment variables if available.

    Expects environment variable 'faissx_API_KEYS' in format:
    "key1:tenant1,key2:tenant2"

    Updates the global API_KEYS dictionary with parsed key-tenant pairs.
    Handles errors gracefully by printing error message if parsing fails.
    """
    env_keys = os.environ.get("faissx_API_KEYS")
    if env_keys:
        try:
            # Parse comma-separated key:tenant pairs
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
        api_key: API key from request header

    Returns:
        Optional[str]: Tenant ID for the API key or None if invalid
    """
    return API_KEYS.get(api_key)


def validate_tenant_access(tenant_id: str, resource_tenant_id: str) -> bool:
    """
    Validate that the tenant has access to the resource.

    Simple equality check - tenant can only access their own resources.
    In a more complex system, this could implement role-based access control.

    Args:
        tenant_id: Tenant ID from API key
        resource_tenant_id: Tenant ID of the resource being accessed

    Returns:
        bool: True if tenant has access, False otherwise
    """
    return tenant_id == resource_tenant_id


class AuthError(Exception):
    """
    Custom exception for authentication failures.
    Raised when an invalid API key is provided.
    """
    pass


class PermissionError(Exception):
    """
    Custom exception for permission failures.
    Raised when a tenant attempts to access resources they don't own.
    """
    pass


def authenticate_request(api_key: str, resource_tenant_id: Optional[str] = None):
    """
    Authenticate a request and validate tenant access if applicable.

    Two-step process:
    1. Validate API key and get tenant ID
    2. If resource_tenant_id provided, verify tenant has access to resource

    Args:
        api_key: API key from request
        resource_tenant_id: Optional tenant ID of the resource being accessed

    Returns:
        str: Tenant ID if authentication successful

    Raises:
        AuthError: If API key is invalid
        PermissionError: If tenant does not have access to the resource
    """
    # Step 1: Validate API key
    tenant_id = get_tenant_id(api_key)
    if tenant_id is None:
        raise AuthError("Invalid API key")

    # Step 2: Validate resource access if resource_tenant_id provided
    if resource_tenant_id is not None and not validate_tenant_access(
        tenant_id, resource_tenant_id
    ):
        raise PermissionError(
            f"Tenant {tenant_id} does not have access to resource owned by {resource_tenant_id}"
        )

    return tenant_id
