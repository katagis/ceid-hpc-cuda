#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_error_macros.h"
#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <initializer_list>

float cublassTime = 0.f;
float basicTime = 0.f;
float optimisedTime = 0.f;

// Returns milliseconds for the time Function f took.
template<typename Function>
float CountTime(Function&& f) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start); CE;
	cudaEventCreate(&stop); CE;

	cudaEventRecord(start); CE;

	f();

	cudaEventRecord(stop); CE;
	cudaEventSynchronize(stop); CE;

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop); CE;
	return milliseconds;
}

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

	cublassTime = CountTime([&]() {
		status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, M, &alpha,
							 inputMatrix.dev_data, M, inputMatrix.dev_data, M, &beta, result.dev_data, N);
	});
	CBE(status);
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

	for (int i = 0; i < nr_rows; ++i) {
		sum += src[AT(i, row, nr_cols)] * src[AT(i, col, nr_cols)];
	}

	result[AT(row, col, nr_cols)] = sum;
}

Matrix basic_Tmutliply(Matrix& input) {
	Matrix result(input.cols, input.cols);
	result.AllocDevice();

	input.IntoDevMatrix();

	basicTime = CountTime([&]() {
		basic_dev_Tmultiply << <1, result.Size() >> > (input.rows, input.cols, input.dev_data, result.dev_data);
	});

	result.FromDevMatrix();

	input.FreeDevice();
	result.FreeDevice();

	return result;
}

Matrix opt_Tmutliply(Matrix& input);

Matrix GetRandomInputMatrix() {
	constexpr int TestRows = 525;
	constexpr int TestCols = 30;

	Matrix inputMatrix(TestCols, TestRows);
	inputMatrix.AllocHost();

	for (int i = 0; i < inputMatrix.Size(); ++i) {
		inputMatrix.data[i] = std::rand() % 64;
	}


	return std::move(inputMatrix);
}

Matrix GetCustomInputMatrix() {
	return std::move(Matrix(4, {
		0,  1,  2,  3,
		10, 11, 12, 13,
		20, 21, 22, 23
	}));
}

int main() {
//	Matrix inputMatrix = GetCustomInputMatrix();
	Matrix inputMatrix = GetRandomInputMatrix();

	cudaSetDevice(0); CE;

	Matrix cublasResult;
	cublasResult = cublas_Tmultiply(inputMatrix);
	
	Matrix basicCudaResult;
	basicCudaResult = basic_Tmutliply(inputMatrix);

	if (cublasResult.IsNearlyEqual(basicCudaResult)) {
		printf("Result: \n");
		//cublasResult.Print();
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
		printf("Opt cuda was correct.\n");
	}
	else {
		printf("Opt cuda was different:\n");
		optCudaResult.Print();
	}


	printf("cuBLASS: %4.4f ms\n", cublassTime);
	printf("basic  : %4.4f ms\n", basicTime);
	printf("optimis: %4.4f ms\n", optimisedTime);


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

__device__ void dev_WriteResult(int nr_rows, int nr_cols, float* src, float* result, int row, int col)
{
	float sum = 0;

	for (int i = 0; i < nr_rows; ++i) {
		sum += src[AT(i, row, nr_cols)] * src[AT(i, col, nr_cols)];
	}

	result[AT(row, col, nr_cols)] = sum;
}


enum class DebugOutput {
	Result = 0,
	RowCol = 1,
	Thread = 2,
	Block = 3
};

constexpr int blocksPerThread = 3;
constexpr DebugOutput debugOutput = DebugOutput::Result;

__global__ void opt_dev_Tmultiply(int nr_rows, int nr_cols, float* src, float* result)
{
	int row = threadIdx.x * blocksPerThread + blockIdx.x;
	int col = threadIdx.y * blocksPerThread + blockIdx.y;
	
	dev_WriteResult(nr_rows, nr_cols, src, result, row, col);


	if (debugOutput == DebugOutput::Result) {

	}
	else if (debugOutput == DebugOutput::RowCol) {
		result[AT(row, col, nr_cols)] = row + col * 100;
	}
	else if (debugOutput == DebugOutput::Thread) {
		result[AT(row, col, nr_cols)] = threadIdx.x + threadIdx.y * 100;
	}
	else if (debugOutput == DebugOutput::Block) {
		result[AT(row, col, nr_cols)] =  blockIdx.x + blockIdx.y * 100;
	}

}

Matrix opt_Tmutliply(Matrix& input) {
	Matrix result(input.cols, input.cols);
	result.AllocDevice();

	input.IntoDevMatrix();


	int totalSide = input.cols;
	int threads = totalSide / blocksPerThread;

	dim3 dimBlocks(blocksPerThread, blocksPerThread);
	dim3 dimThreads(threads, threads);
	

	optimisedTime = CountTime([&]() {

		// vecAdd <<<numOfBlocks, threadsPerBlock>>>

		opt_dev_Tmultiply << <dimBlocks, dimThreads >> > (input.rows, input.cols, input.dev_data, result.dev_data);
	});
	result.FromDevMatrix();

	input.FreeDevice();
	result.FreeDevice();

	return result;
}
