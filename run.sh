#!/bin/bash -l

# Remove the module loading lines since they are not available on your local machine
# module load nvcc cuda
 
rm -rf ./output

# Navigate to the cuda directory
cd cuda

# Clean previous build files
make clean

# Build the project
make

# Run the executable
cd ..
# Create output directory if it doesn't exist
mkdir -p output
./MolecularDynamics
