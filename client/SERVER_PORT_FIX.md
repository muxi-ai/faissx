# Server Port Parameter Fix

## Issue Description

The FAISSx server was ignoring the port parameter when launched directly using the server module. This caused:

1. When running `python -m faissx.server.server --port 45679`, the server would still bind to port 45678
2. The `run_remote_tests.sh` script could fail if the default port was already in use
3. Inconsistent behavior between CLI and direct module execution

## Changes Made

### 1. Fixed `faissx/server/server.py`

- Added proper command-line argument parsing with `argparse`
- Replaced the environment variable lookup `os.environ.get("faissx_PORT", DEFAULT_PORT)` with command-line arguments
- Added `--port`, `--bind-address`, and `--data-dir` command-line parameters
- Preserved the environment variable fallback but prioritized command-line arguments
- Fixed environment variable name inconsistencies

### 2. Updated `client/run_remote_tests.sh`

- Changed to use the server module directly instead of a wrapper script
- Modified the port to use 45679 by default to avoid conflicts
- Added proper port configuration via environment variable
- Fixed Python import paths to correctly load test modules
- Ensured test script uses the correct server port

### 3. Updated `client/test_remote_no_fallback.py`

- Added support for configurable port via environment variable
- Replaced hardcoded server URL with a dynamic one that respects the port setting
- Maintained backward compatibility with default port

## Testing

The changes were verified by:

1. Running the server with a custom port: `python -m faissx.server.server --port 45679`
2. Checking that it correctly listens on the specified port
3. Connecting to the custom port with a FAISSx client
4. Running tests against the custom port server
5. Running the complete test suite with the updated scripts

## Results

- The server now correctly binds to the specified port
- Tests run successfully against the custom port
- The run_remote_tests.sh script correctly starts and stops the server on the specified port
- All tests pass, confirming the proper functionality of remote mode with no fallback
