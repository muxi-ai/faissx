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

This module handles API key authentication and tenant isolation for the FAISSx server.
It provides functions for managing API keys, loading credentials, validating requests,
enforcing tenant-level access control, and handling auth/permission failures.

The authentication system ensures multi-tenant security by restricting clients to
only access indices and vectors belonging to their assigned tenant.
"""

import os
from typing import Dict, Optional
import logging

logger = logging.getLogger("faissx.server")

# In-memory storage for API key to tenant ID mappings
# TODO: Replace with database or config file in production
API_KEYS: Dict[str, str] = {}


def set_api_keys(keys: Dict[str, str]):
    """
    Initialize the global API_KEYS dictionary with provided key-tenant mappings.

    Used by server.configure to set up API keys. Creates a defensive copy to
    prevent external modifications to the internal state.

    Args:
        keys: Dictionary mapping API keys to their corresponding tenant IDs
    """
    global API_KEYS
    API_KEYS = keys.copy() if keys else {}


def load_api_keys_from_env():
    """
    Load API key configurations from environment variables.

    Expects 'faissx_API_KEYS' env var in format: "key1:tenant1,key2:tenant2"
    Parses the string into key-tenant pairs and updates the global API_KEYS dict.
    Gracefully handles parsing errors by logging them.
    """
    env_keys = os.environ.get("faissx_API_KEYS")
    if env_keys:
        try:
            # Split into individual key:tenant pairs and process each
            pairs = env_keys.split(",")
            for pair in pairs:
                key, tenant = pair.split(":")
                API_KEYS[key.strip()] = tenant.strip()
        except Exception as e:
            logger.critical(f"Error loading API keys from environment: {e}")


def get_tenant_id(api_key: str) -> Optional[str]:
    """
    Retrieve the tenant ID associated with a given API key.

    Args:
        api_key: The API key to look up from request headers

    Returns:
        The tenant ID if the key exists, None otherwise
    """
    return API_KEYS.get(api_key)


def validate_tenant_access(tenant_id: str, resource_tenant_id: str) -> bool:
    """
    Verify if a tenant has permission to access a specific resource.

    Currently implements simple ownership check - tenants can only access their
    own resources. Could be extended to support role-based access control.

    Args:
        tenant_id: The tenant ID from the API key
        resource_tenant_id: The tenant ID of the resource being accessed

    Returns:
        True if access is allowed, False otherwise
    """
    return tenant_id == resource_tenant_id


class AuthError(Exception):
    """
    Exception raised when API key validation fails.
    Indicates the provided API key is invalid or missing.
    """

    pass


class PermissionError(Exception):
    """
    Exception raised when a tenant attempts unauthorized resource access.
    Indicates the tenant lacks permission to access the requested resource.
    """

    pass


def authenticate_request(api_key: str, resource_tenant_id: Optional[str] = None):
    """
    Perform two-step authentication and authorization check.

    1. Validates the API key and retrieves associated tenant ID
    2. If resource_tenant_id is provided, verifies tenant has access rights

    Args:
        api_key: The API key from the request
        resource_tenant_id: Optional tenant ID of the resource being accessed

    Returns:
        The authenticated tenant ID

    Raises:
        AuthError: When API key is invalid
        PermissionError: When tenant lacks resource access permission
    """
    # First step: Validate API key and get tenant ID
    tenant_id = get_tenant_id(api_key)
    if tenant_id is None:
        raise AuthError("Invalid API key")

    # Second step: Check resource access permissions if applicable
    if resource_tenant_id is not None and not validate_tenant_access(
        tenant_id, resource_tenant_id
    ):
        raise PermissionError(
            f"Tenant {tenant_id} does not have access to resource owned by {resource_tenant_id}"
        )

    return tenant_id
