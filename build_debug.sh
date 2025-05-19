#!/bin/bash

# Debug script to find and test copy operations

set -e
set -x

# Create test directories
mkdir -p test_output

echo "Current directory: $(pwd)"

# Look for binary in build directory
echo "Looking for binaries..."

# PyOxidizer creates binaries in various locations
POSSIBLE_PATHS=(
  "./build"
  "/var/folders/*/T/pyoxidizer*/build"
  "/tmp/*/build"
  "/tmp/pyoxidizer*/build"
  "~/Library/Caches/pyoxidizer"
)

for path in "${POSSIBLE_PATHS[@]}"; do
  echo "Searching in: $path"
  if [ -d "$path" ]; then
    echo "Directory exists: $path"
    find "$path" -type f -name "faiss-proxy-server" 2>/dev/null || echo "No binaries found"
  else
    echo "Directory does not exist: $path"
  fi
done

# Create a test file and copy it
echo "Creating and copying test file..."
echo "Test content" > test_file.txt
cp -v test_file.txt test_output/
ls -la test_output/

# Clean up
rm test_file.txt
rm -rf test_output

echo "Debug completed"
