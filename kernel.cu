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

Matrix GetRandomInputMatrix() {
	constexpr int TestRows = 1601;
	constexpr int TestCols = 1601;

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

// Integer division + ceil if remainder
int div_ceil(int lhs, int rhs) {
	return lhs / rhs + (lhs % rhs == 0 ? 0 : 1);
}

Matrix cublas_Tmultiply(Matrix& inputMatrix) {
	const double alpha = 1;
	const double beta = 0;

	Matrix result(inputMatrix.cols, inputMatrix.cols);
	result.AllocDevice();

	inputMatrix.IntoDevMatrix_ColMajor();
	
	cublasHandle_t handle;
	auto status = cublasCreate(&handle); CBE(status);

	int N = inputMatrix.cols;
	int M = inputMatrix.rows;

	cublassTime = CountTime([&]() {
		status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, M, &alpha,
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
__global__ void basic_dev_Tmultiply(int nr_rows, int nr_cols, double* src, double* result)
{
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= nr_cols || col >= nr_cols) {
		return;
	}
	
	double elem = 0;

	for (int i = 0; i < nr_rows; ++i) {
		elem += src[AT(i, row, nr_cols)] * src[AT(i, col, nr_cols)];
	}

	result[AT(row, col, nr_cols)] = elem;

}

Matrix basic_Tmutliply(Matrix& input) {
	Matrix result(input.cols, input.cols);
	result.AllocDevice();

	input.IntoDevMatrix();

	constexpr int Threads = 32;
	const int GridSize = div_ceil(input.cols, Threads);

	dim3 block(Threads, Threads);
	dim3 grid(GridSize, GridSize);

	basicTime = CountTime([&]() {
		basic_dev_Tmultiply << <grid, block>> > (input.rows, input.cols, input.dev_data, result.dev_data);
	});

	result.FromDevMatrix();

	input.FreeDevice();
	result.FreeDevice();

	return result;
}

Matrix opt_Tmutliply(Matrix& input);


int main() {
//	Matrix inputMatrix = GetCustomInputMatrix();
	Matrix inputMatrix = GetRandomInputMatrix();

	cudaSetDevice(0); CE;

	Matrix cublasResult;
	cublasResult = cublas_Tmultiply(inputMatrix);
	printf("Result: \n");

	Matrix basicCudaResult;
	basicCudaResult = basic_Tmutliply(inputMatrix);

	if (!cublasResult.IsNearlyEqual(basicCudaResult)) {
		printf("basic cuda had different result than cublass");
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

enum class DebugOutput {
	Result = 0,
	RowCol = 1,
	Thread = 2,
	Block = 3
};

constexpr DebugOutput debugOutput = DebugOutput::Result;

constexpr int BLOCK_SIZE = 32;
constexpr int TILE_SIZE = 32;

__device__ void dev_WriteResult(int nr_rows, int nr_cols, double* src, double* result, int row, int col)
{
	double sum = 0;

	for (int i = 0; i < nr_rows; ++i) {
		sum += src[AT(i, row, nr_cols)] * src[AT(i, col, nr_cols)];
	}

	result[AT(row, col, nr_cols)] = sum;
	result[AT(col, row, nr_cols)] = sum;
}




__global__ void opt_dev_Tmultiply(int nr_rows, int nr_cols, double* src, double* result)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	
	// Triangular due to symmetry of t(A) * A && border checks
	if (row > col || col >= nr_cols) {
		return;
	}


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


	constexpr int Threads = BLOCK_SIZE;
	const int GridSize = div_ceil(input.cols, TILE_SIZE);

	dim3 block(Threads, Threads);
	dim3 grid(GridSize, GridSize);
	

	optimisedTime = CountTime([&]() {
		opt_dev_Tmultiply <<<grid, block>>> (input.rows, input.cols, input.dev_data, result.dev_data);
	});
	result.FromDevMatrix();

	input.FreeDevice();
	result.FreeDevice();

	return result;
}
