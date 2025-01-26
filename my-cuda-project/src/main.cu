#include <cuda_runtime.h>
#include "kernels.h"

int main() {
    // Launch the dummy kernel
    dummyKernel<<<2,4>>>();

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    return 0;
}

