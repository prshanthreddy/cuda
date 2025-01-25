#include <stdio.h>
#include <cuda_runtime.h>

// __global__ means this function is called from the host and executed on the device
__global__ void helloGPU() {
    printf("Hello from GPU!\n");
}

int main() {
    helloGPU<<<1, 1>>>(); // <<<blocks, threads>>> syntax is used to launch the kernel
    
    // Wait for the kernel to finish
    cudaDeviceSynchronize(); // Without this, the program may exit before the kernel finishes
    
    printf("Hello from CPU!\n");
    return 0;
}

