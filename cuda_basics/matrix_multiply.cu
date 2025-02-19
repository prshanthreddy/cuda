#include <cuda_runtime.h>
#include <stdio.h>

// Tile size is the size of the shared memory tile used to store the sub-matrix of A and B that are used to compute the sub-matrix of C
#define TILE_SIZE 16

// Define a global kernel function that multiplies two matrices

