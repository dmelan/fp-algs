#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 file_name"
    exit 1
fi

OUTPUT_DIR="data"
mkdir -p "$OUTPUT_DIR"

file_name="$1"
FreeFem++ "$file_name" 2>&1