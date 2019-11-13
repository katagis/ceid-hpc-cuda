
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_error_macros.h"

#include <stdio.h>

void addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

	CE_SETUP();

    // Add vectors in parallel.
    addWithCuda(c, a, b, arraySize) CE;
    
    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaDeviceReset() CE;

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
void addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;

	CE_SETUP(dev_a, dev_b, dev_c);


	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaSetDevice(0) CE;

	// Allocate GPU buffers for three vectors (two input, one output).
	cudaMalloc((void**)&dev_c, size * sizeof(int)) CE;

	cudaMalloc((void**)&dev_a, size * sizeof(int)) CE;
	cudaMalloc((void**)&dev_b, size * sizeof(int)) CE;


	// Copy input vectors from host memory to GPU buffers.
	cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice) CE;
	cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice) CE;

	// Launch a kernel on the GPU with one thread for each element.
	addKernel <<<1, size >>> (dev_c, dev_a, dev_b) CE;

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaDeviceSynchronize() CE;

	// Copy output vector from GPU buffer to host memory.
	cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost) CE;
}
