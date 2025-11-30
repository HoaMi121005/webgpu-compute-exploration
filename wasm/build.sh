#!/bin/bash

echo "Building WASM module..."

if ! command -v wasm-pack &> /dev/null; then
    echo "Error: wasm-pack not found. Install with: cargo install wasm-pack"
    exit 1
fi

wasm-pack build --target web --out-dir ../js/wasm-pkg

echo "Build complete! WASM module is ready at js/wasm-pkg/"
