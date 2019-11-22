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


Matrix GetRandomInputMatrix();

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
#define AT_T(row, col, nr_rows) (row + col * nr_rows)


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
		printf("cublas:\n");
		cublasResult.Print();

		printf("src:\n");
		inputMatrix.Print();

		printf("cuda:\n");
		optCudaResult.Print();
	}


	printf("cuBLASS: %4.4f ms\n", cublassTime);
	printf("basic  : %4.4f ms\n", basicTime);
	printf("optimis: %4.4f ms\n", optimisedTime);


	cudaDeviceReset();
	return 0;
}

Matrix GetRandomInputMatrix() {
	constexpr int TestRows = 5120;
	constexpr int TestCols = 2560;

	Matrix inputMatrix(TestCols, TestRows);
	inputMatrix.AllocHost();

	for (int i = 0; i < inputMatrix.Size(); ++i) {
		inputMatrix.data[i] = i % TestCols;
	}

	for (int i = 0; i < TestRows; ++i) {
		inputMatrix.data[i] = i * 10;
	}


	return std::move(inputMatrix);
}


//
// OPTIMISED CUDA
//

//
// Optimisation Notes: 
// 1. T(A) * A is symmetrical
// 2. 
//

// Optimisations done based on:
// https://www.seas.upenn.edu/~cis565/Lectures2011S/Lecture12.pdf // Prefetch 
// https://ecatue.gitlab.io/gpu2018/pages/Cookbook/matrix_multiplication_cuda.html#5 // Shared memory + Bank conflicts

enum class DebugOutput {
	Result = 0,
	RowCol = 1,
	Thread = 2,
	Block = 3
};

constexpr DebugOutput debugOutput = DebugOutput::Result;

constexpr int BLOCK_SIZE = 32;
constexpr int TILE_SIZE = 32;

__device__ void dev_WriteResult(int nr_rows, int nr_cols, double* src, double* output, int res_row, int res_col)
{
	const int bx = blockIdx.x;
	const int by = blockIdx.y;

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	const int block_start_x = bx * TILE_SIZE;
	const int block_start_y = by * TILE_SIZE;

	// transpose(A) * A in practice means we multiply between 2 columns. With this as the base assumption the rest of
	// the code in this function may use the terms column, column_1 and column_2 meaning the columns from the "src"
	// that are needed to be multiplyied. This makes the code simpler and easier to read.
	
	// Also note that A is in Column-Major

	__shared__ double sm_col_onY[TILE_SIZE][TILE_SIZE];
	__shared__ double sm_col_onX[TILE_SIZE][TILE_SIZE];

	double result = 0.0;


	// prepare first loop data...

	// Column to copy for this thread based on the C tile's x
	const int col_x = block_start_x + tx;
	
	// Column to copy for this thread based on the C tile's y
	const int col_y = block_start_y + tx;

	int m;
	for (m = 0; m < nr_rows; m += TILE_SIZE) {

		// Copying here is a bit different than the usual A*B multiplication
		// We copy blocks from the required columns directly as squares
		sm_col_onX[ty][tx] = src[AT_T(m + ty, col_x, nr_rows)];
		sm_col_onY[ty][tx] = src[AT_T(m + ty, col_y, nr_rows)];

		__syncthreads();

#pragma unroll
		for (int k = 0; k < TILE_SIZE; ++k) {
			result += sm_col_onX[k][tx] * sm_col_onY[k][ty];
		}


		__syncthreads();
	}

	output[AT(res_row, res_col, nr_cols)] = result;
	output[AT(res_col, res_row, nr_cols)] = result;
}



__global__ void opt_dev_Tmultiply(int nr_rows, int nr_cols, double* src, double* result)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (blockIdx.x > blockIdx.y) {
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

	input.IntoDevMatrix_ColMajor();


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
