#!/bin/bash
set -e

echo "=== Cleaning up PyOxidizer artifacts and temporary files ==="

# Remove build artifacts
echo "Removing build artifacts..."
rm -rf server/build/
rm -rf server/build_temp/
rm -rf server/dist/
rm -rf dist/
rm -rf server/temp_project/

# Remove temporary MD files
echo "Removing temporary documentation files..."
rm -f IMPLEMENTATION_STATUS.md
rm -f ZMQ_IMPLEMENTATION_SUMMARY.md
rm -f ZMQ_MIGRATION_PLAN.md
rm -f NEXT_STEPS.md

# Remove PyOxidizer configuration
echo "Removing PyOxidizer configuration..."
rm -f pyoxidizer.bzl
rm -f server/pyoxidizer.bzl

# Remove build scripts
echo "Removing build scripts..."
rm -f build_macos_arm64.sh

# Keep the server.py and Docker files
echo "Keeping server.py and Docker setup which is working"

echo "=== Cleanup complete ==="
