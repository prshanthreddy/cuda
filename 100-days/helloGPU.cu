#include <stdio.h>
#include <cuda_runtime.h>

// __global__ function to be executed on the GPU which is a kernel function

__global__ void helloGPU() {
    if (threadIdx.x == 0) {
        printf("Hello from GPU thread %d!\n", threadIdx.x);
    }
}

// Main function to launch the kernel

int main() {
    // Launch the kernel with 1 block and 1 thread
    helloGPU<<<1, 1>>>();

    // Wait for the GPU to finish before accessing the results
    cudaDeviceSynchronize();

    // Check for errors in kernel launch (optional)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error launching kernel: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}