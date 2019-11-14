#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_error_macros.h"
#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <initializer_list>

Matrix cublas_Tmultiply(Matrix& inputMatrix) {
	const float alpha = 1;
	const float beta = 0;

	Matrix result(inputMatrix.cols, inputMatrix.cols);
	result.AllocDevice();

	inputMatrix.IntoDevMatrix_ColMajor();
	
	cublasHandle_t handle;
	auto status = cublasCreate(&handle); CBE(status);

	int N = inputMatrix.cols;
	int M = inputMatrix.rows;
	status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, M, &alpha,
							  inputMatrix.dev_data, M, inputMatrix.dev_data, M, &beta, result.dev_data, N); CBE(status);
	cublasDestroy(handle);


	result.FromDevMatrix_ColMajor();

	result.FreeDevice();
	inputMatrix.FreeDevice();

	return std::move(result);
}

#define AT(row, col, nr_cols) (row * nr_cols + col)


//
// SIMPLE CUDA
//
__global__ void basic_dev_Tmultiply(int nr_rows, int nr_cols, float* src, float* result)
{
	int row = threadIdx.x / nr_cols;
	int col = threadIdx.x % nr_cols;
	
	float sum = 0;

	for (int i = 0; i < nr_cols; ++i) {
		sum += src[AT(i, row, nr_cols)] * src[AT(i, col, nr_cols)];
	}

	result[AT(row, col, nr_cols)] = sum;

//	Index Debugger
//	result[AT(row, col, nr_cols)] = row + 10 * col;
}

Matrix basic_Tmutliply(Matrix& input) {
	Matrix result(input.cols, input.cols);
	result.AllocDevice();

	input.IntoDevMatrix();

	basic_dev_Tmultiply<<<1, result.Size()>>>(input.rows, input.cols, input.dev_data, result.dev_data);

	result.FromDevMatrix();

	input.FreeDevice();
	result.FreeDevice();

	return result;
}




Matrix opt_Tmutliply(Matrix& input);

int main() {
	Matrix inputMatrix(4, {
		0,  1,  2,  3,
		10, 11, 12, 13,
		20, 21, 22, 23
	});

	cudaSetDevice(0); CE;

	Matrix cublasResult;
	cublasResult = cublas_Tmultiply(inputMatrix);
	
	Matrix basicCudaResult;
	basicCudaResult = basic_Tmutliply(inputMatrix);

	if (cublasResult.IsNearlyEqual(basicCudaResult)) {
		printf("Result: \n");
		cublasResult.Print();
	}
	else {
		printf("Different results: \ncublas:\n");
		cublasResult.Print();
		printf("simple cuda:\n");
		basicCudaResult.Print();
	}

	Matrix optCudaResult;
	optCudaResult = opt_Tmutliply(inputMatrix);

	if (cublasResult.IsNearlyEqual(optCudaResult)) {
		printf("Opt cuda was correct!\n");
	}
	else {
		printf("Opt cuda was different:\n");
		optCudaResult.Print();
	}

	cudaDeviceReset();
	return 0;
}


//
// OPTIMISED CUDA
//

//
// Optimisation Notes: 
// 1. T(A) * A is symmetrical
// 2. 
//
//
// Testing:
// 1. Fill our 11GB GPU buffer without crashing the the OS & driver. 
//    Check if offsets can be applied to make 1 buffer with multiple cuda memcopies.
// 

__global__ void opt_dev_Tmultiply(int nr_rows, int nr_cols, float* src, float* result)
{
	int row = threadIdx.x / nr_cols;
	int col = threadIdx.x % nr_cols;

	float sum = 0;

	for (int i = 0; i < nr_cols; ++i) {
		sum += src[AT(i, row, nr_cols)] * src[AT(i, col, nr_cols)];
	}

	result[AT(row, col, nr_cols)] = sum;

	//	Index Debugger
	//	result[AT(row, col, nr_cols)] = row + 10 * col;
}

Matrix opt_Tmutliply(Matrix& input) {
	Matrix result(input.cols, input.cols);
	result.AllocDevice();

	input.IntoDevMatrix();

	basic_dev_Tmultiply << <1, result.Size() >> > (input.rows, input.cols, input.dev_data, result.dev_data);

	result.FromDevMatrix();

	input.FreeDevice();
	result.FreeDevice();

	return result;
}
