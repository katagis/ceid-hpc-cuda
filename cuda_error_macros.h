#pragma once
#include "cuda_runtime.h"
#include <stdio.h>
#include "cublas_v2.h"


// Text paste helper macro
#define CE_TEXT_PASTE(X,Y) X##Y 
#define CE_COMBINE(X,Y) CE_TEXT_PASTE(X,Y)
#define CE_LINEVAR CE_COMBINE(Z_status_, __LINE__)

// Use after each cuda call you want to check for errors. Starts with a ';'. Intended example usage is: "cudaSetDevice(0) CE;"
#define CE ; cudaError_t CE_LINEVAR = cudaGetLastError(); \
if (CE_LINEVAR != cudaSuccess) { \
	fprintf(stderr, "Cuda failure at line %d: [%d] %s\n", \
			__LINE__, \
			CE_LINEVAR, \
			cudaGetErrorString(CE_LINEVAR) \
	); \
	cudaDeviceReset(); \
	exit(1); \
} do{}while(0)



#define CBE(STATUS) ; \
if (STATUS != cudaSuccess) { \
	fprintf(stderr, "cuBLASS failure at line %d: [%d] %s\n", \
			__LINE__, \
			STATUS, \
			cublasGetErrorString(STATUS) \
	); \
	cudaDeviceReset(); \
	exit(1); \
} do{}while(0)

const char* cublasGetErrorString(cublasStatus_t status)
{
	switch (status)
	{
	case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
	case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
	case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
	case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
	case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
	case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
	case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
	case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
	}
	return "unknown error";
}
