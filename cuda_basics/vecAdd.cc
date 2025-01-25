#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vecAdd(float* A, float* B, float* C) {
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main(){
    vecAdd<<<1, 10>>>(A, B, C);
    return 0;
    
}