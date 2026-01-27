#!/bin/bash
# Command-line interface examples for Molecular Entropy

# Basic usage
molecular-entropy \
    --pdb complex.pdb \
    --chains A,B \
    --rotlib rotamer_library/simple_library.parquet

# Specify output directory
molecular-entropy \
    --pdb complex.pdb \
    --chains A,B \
    --rotlib rotamer_library/simple_library.parquet \
    --out results/

# Custom temperature (physiological)
molecular-entropy \
    --pdb complex.pdb \
    --chains A,B \
    --rotlib rotamer_library/simple_library.parquet \
    --T 310.15

# Custom translational/rotational penalty
molecular-entropy \
    --pdb complex.pdb \
    --chains A,B \
    --rotlib rotamer_library/simple_library.parquet \
    --tr-penalty 10.0

# Skip specific components for faster calculation
molecular-entropy \
    --pdb complex.pdb \
    --chains A,B \
    --no-anm \
    --no-rotamer

# Verbose output for debugging
molecular-entropy \
    --pdb complex.pdb \
    --chains A,B \
    --rotlib rotamer_library/simple_library.parquet \
    -v

# Process multiple files with a loop
for pdb in structures/*.pdb; do
    name=$(basename "$pdb" .pdb)
    molecular-entropy \
        --pdb "$pdb" \
        --chains A,B \
        --rotlib rotamer_library/simple_library.parquet \
        --out "results/${name}/"
done
