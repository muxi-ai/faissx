# Changes to FAISSx Client Fallback Behavior

This document summarizes the changes made to fix the fallback-to-local behavior in the FAISSx client.

## Problem Statement

Previously, when a FAISSx client was configured to use remote mode but encountered issues with the server, it would silently fall back to local mode. This behavior was problematic because:

1. Data added to a remote index would later be silently searched locally when server issues occurred
2. Different index types behaved inconsistently regarding fallback:
   - IndexFlatL2 worked properly in remote mode
   - IndexIDMap explicitly raised errors about not being implemented
   - IndexIVFFlat, IndexHNSWFlat, IndexPQ, and IndexScalarQuantizer silently fell back to local mode
3. The inconsistent fallback behavior led to confusion and potential data inconsistency

## Changes Made

### 1. Updated `get_client()` in client.py

- Modified to retry connections to the server a few times
- When a server URL is configured but connection fails, it now raises a clear error instead of returning None
- Never falls back to local mode when remote mode is explicitly configured

### 2. Updated Index Implementations

#### IndexIDMap

- Added proper remote mode implementation with support for all operations
- Added server-side implementation for IDMap and IDMap2 indices
- Implemented add_with_ids, remove_ids, reconstruct, and replace_vector operations on the server
- Added full client-side support for IDMap operations in remote mode

#### IndexIVFFlat

- Added proper error checking for server connections
- Added explicit exception handling for the `create_index` call
- Removed the fallback-to-local behavior, raising clear errors instead

#### IndexHNSWFlat

- Added proper error checking for server connections
- Added explicit exception handling for the `create_index` call
- Removed the fallback-to-local behavior, raising clear errors instead

#### IndexPQ

- Added proper error checking for server connections
- Added explicit exception handling for the `create_index` call
- Removed the fallback-to-local behavior, raising clear errors instead

#### IndexScalarQuantizer

- Added proper error checking for server connections
- Added explicit exception handling for the `create_index` call
- Removed the fallback-to-local behavior, raising clear errors instead

### 3. Added Testing Utilities

- Created `test_remote_no_fallback.py` to verify proper behavior of different index types in remote mode
- Created `test_idmap_remote.py` to test the IDMap implementation in remote mode
- Created `run_remote_tests.sh` to automate running the server and tests together
- Tests validate that:
  - IndexFlatL2 works properly in remote mode
  - IndexIVFFlat, IndexHNSWFlat, IndexPQ, and IndexScalarQuantizer either work properly or raise clear errors when not implemented
  - IndexIDMap and IndexIDMap2 work properly in remote mode
  - When server is unavailable, clear connection errors are raised

### 4. Fixed Port Parameter Handling

- Added proper command-line argument parsing for server port
- Fixed issue where the server would ignore the specified port
- Updated test infrastructure to use configurable ports

## Expected Behavior

After these changes, the FAISSx client should:

1. Operate in either remote mode OR local mode, never silently switching between them
2. When remote operations fail:
   - Retry a few times
   - Raise clear errors if retries fail
   - Never fall back to local mode
3. Provide consistent error reporting across all index types
4. Support all IndexIDMap operations in remote mode
5. Allow configuring the server port from command line

## Remaining Work

The server implementation for remote mode is incomplete for some index types:

1. The `create_index` method has issues with various specialized index types on the server side

These server-side issues should be addressed in future updates, but the client now properly handles these cases by raising clear errors rather than silently falling back to local mode.
