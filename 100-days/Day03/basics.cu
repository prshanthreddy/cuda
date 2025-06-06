#include<iostream>
#include<cuda_runtime.h>

using namespace std;

// defining my kernel function
// Vector addition kernel
__global__ void myKernel(float *a, float *b, float *c, int n) {
    // Get the thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Check if the index is within bounds
    if (idx < n) {
        // Perform some operation, e.g., addition
        c[idx] = a[idx] + b[idx];
    }
    return;
}

// Vector subtraction kernel
__global__ void myKernelSub(float *a, float *b, float *c, int n) {
    // Get the thread index
    int idx=threadIdx.x+ blockIdx.x * blockDim.x;
    // Check if the index is within bounds
    if (idx < n) {
        // Perform some operation, e.g., subtraction
        c[idx] = b[idx] - a[idx];
    }

}
// Addition of two arrays 
int main(){
    //declare the array size
    int n=1000;
    size_t size = n * sizeof(int);
    // declare the host arrays
    float *h_a, *h_b, *h_c;
    // allocate memory for the host arrays
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);
    // initialize the host arrays
    for(int i = 0; i < n; i++) {
        h_a[i] = static_cast<float>(i);     // For example values will be 0, 1, 2, ..., 999
        h_b[i] = static_cast<float>(i * 2); // For example values will be 0, 2, 4, ..., 1998
        h_c[i] = 0.0f;

    }
    //allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);
    // copy the host arrays to the device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    // define the number of threads per block and the number of blocks
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock; // round up to the nearest whole block
    // launch the kernel with 1 block and n threads
    myKernel<<<blocksPerGrid, threadsPerBlock>>>(  d_a, d_b, d_c,n);

    // copy the result back to the host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    // print the result
    cout << "Result of addition:" << endl;
    for(int i = 0; i < n; i++) {
        cout << "Index " << i << ": ";
        cout<<h_a[i] << " + " << h_b[i] << " = " << h_c[i] << endl;
    }
    cout << endl;

    // launch the kernel with 1 block and n threads for subtraction
    myKernelSub<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    // copy the result back to the host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    // print the result of subtraction
    cout << "Result of subtraction:" << endl;
    for(int i = 0; i < n; i++) {
        cout << "Index " << i << ": ";
        cout<<h_b[i] << " - " << h_a[i] << " = " << h_c[i] << endl;
    }

    // free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}