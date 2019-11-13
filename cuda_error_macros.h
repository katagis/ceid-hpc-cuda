#pragma once
#include "cuda_runtime.h"

namespace detail {
template<typename T>
int DoFree(T* ptr, int) {
	cudaFree(ptr);
	return 0;
}

template<typename... T>
int Nothing(T... ptr) {
	return 0;
}
}

template<typename... T>
void CudaMultiFree(T*... args) {
	detail::Nothing(detail::DoFree(args, 0)...);
}

// Provide a list of buffers to cudaFree on error.
#define CE_SETUP(...) cudaError_t Z_cudaStatus; auto onErrorLambda = [__VA_ARGS__]() { CudaMultiFree(__VA_ARGS__); cudaDeviceReset(); exit(1); }
// Use after each cuda call you want to check for errors. Starts with a ';'. Intended example usage is: "cudaSetDevice(0) CE;"
#define CE ; Z_cudaStatus = cudaGetLastError(); if (Z_cudaStatus != cudaSuccess) { fprintf(stderr, "Cuda failure at line %d: [%d] %s\n", __LINE__, Z_cudaStatus, cudaGetErrorString(Z_cudaStatus)); onErrorLambda(); } do{}while(0)
