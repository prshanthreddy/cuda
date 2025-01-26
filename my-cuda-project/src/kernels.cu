#include <stdio.h>
#include "kernels.h"

// A dummy kernel that prints block and thread indices
__global__ void dummyKernel() {
    printf("Block index: %d, Thread index: %d\n", blockIdx.x, threadIdx.x);
}



