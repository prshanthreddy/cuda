#include<iostream>
#include<cuda_runtime.h>
using namespace std;

// Kernel function to show the threadid
__global__ void showGlobalThreadIdx() {
    // Get the global thread index
    int globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    // Print the global thread index
    printf("Global Thread Index: %d, Block Index: %d, Thread Index: %d\n", globalThreadIdx, blockIdx.x, threadIdx.x);
}

int main(){
    //defining the number of threads per block
    int N = 1000;
    int threadsPerBlock = 256;
    //defining the number of blocks
    int numverOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock; // round up to the nearest whole block
    // Launch the kernel with the specified number of blocks and threads per block
    showGlobalThreadIdx<<<numverOfBlocks, threadsPerBlock>>>();
    // Synchronize the device to ensure all threads have completed
    cudaDeviceSynchronize();
    // Check for errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "Error in kernel launch: " << cudaGetErrorString(err) << endl;
        return -1;
    }
    return 0;
}