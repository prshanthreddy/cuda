#include <iostream>

// Define a global kernel function that adds two vectors
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int globalId=blockIdx.x*blockDim.x+threadIdx.x;
    //globalId is the index of the element in the array
    // blockIdx.x is the block index
    // blockDim.x is the number of threads in a block
    if(globalId<N){
        C[globalId]=A[globalId]+B[globalId];
    }
}

int main() {
    const int N = 1000000;
    // Allocate memory on the host where size_t is an unsigned integer type of at least 16 bits
    size_t size = N * sizeof(float);
    // Allocate memory on the host
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    // Initialize host arrays
    for(int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    // Allocate memory on the device
    float *d_a, *d_b,*d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    // Copy host arrays to device
    cudaMemcpy(d_a, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_B, size, cudaMemcpyHostToDevice);
    // Define block and grid sizes
    int blockSize=256;
    int gridSize=(N+blockSize-1)/blockSize;

    // Launch the kernel
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    // Copy the result back to the host
    cudaMemcpy(h_C, d_c, size, cudaMemcpyDeviceToHost);
    // Print the result
    printf("Printing the result\n");
    for(int i = 0; i < 10; i++) {
        std::cout << h_C[i] << std::endl;
    }
    // Verify the result
    printf("Verifying the result\n");
    bool success = true;
    for(int i = 0; i < N; i++) {
        if(h_C[i] != h_A[i] + h_B[i]) {
            success = false;
            std::cout << "Error at index " << i << std::endl;
            break;
        }
    }
    if (success){
        std::cout<<"All elements are correct"<<std::endl;
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}