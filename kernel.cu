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

#define AT(row, col, nr_cols) ((row) * (nr_cols) + (col))
#define AT_T(row, col, nr_rows) ((row) + ((col)* (nr_rows)))


//
// SIMPLE CUDA
//
__global__ void basic_dev_Tmultiply(int nr_rows, int nr_cols, double* src, double* result)
{
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < col || row >= nr_cols || col >= nr_cols) {
		return;
	}
	
	double elem = 0;

	for (int i = 0; i < nr_rows; ++i) {
		elem += src[AT(i, row, nr_cols)] * src[AT(i, col, nr_cols)];
	}

	result[AT(row, col, nr_cols)] = elem;
	result[AT(col, row, nr_cols)] = elem;
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

		printf("cuda:\n");
		optCudaResult.Print();
	}


	printf("cuBLASS: %4.4f ms\n", cublassTime);
	printf("basic  : %4.4f ms\n", basicTime);
	printf("optimis: %4.4f ms\n", optimisedTime);


	cudaDeviceReset();
	return 0;
}

enum class DebugOutput {
	Result = 0,
	RowCol = 1,
	Thread = 2,
	Block = 3
};

constexpr DebugOutput debugOutput = DebugOutput::Result;

constexpr int BLOCK_SIZE = 32;
constexpr int TILE_SIZE = BLOCK_SIZE;


Matrix GetRandomInputMatrix() {
	constexpr int TestRows = 5120;
	constexpr int TestCols = 10240;

	Matrix inputMatrix(TestCols, TestRows);
	inputMatrix.AllocHost();

	for (int i = 0; i < inputMatrix.Size(); ++i) {
		inputMatrix.data[i] = i % TestCols;
	}

	for (int i = 0; i < TestCols; ++i) {
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


// What we tried:
// Registers: no change
// Handle Bx == By with a different function: no speed change
// Reduce overhead by launching half grid for exact triangle: speed got worse
// Thread Granularity: Worse performance due to exceeding our device's Shared Memory Per block at 2x2 / thread or exceeding registers at 2x1 / per thread. 
//                     Required dropping double buffering

// Fixing 

__device__ void dev_WriteResult(int nr_rows, int nr_cols, double* src, double* output, const int res_row, const int res_col)
{

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	const int bx = blockIdx.x;
	const int by = blockIdx.y;

	const int block_start_x = bx * TILE_SIZE;
	const int block_start_y = by * TILE_SIZE;

	// transpose(A) * A in practice means we multiply between 2 columns. With this as the base assumption the rest of
	// the code in this function may use the terms column, column_1 and column_2 meaning the columns from the "src"
	// that are needed to be multiplyied. This makes the code simpler and easier to read.

	// Copying here is a bit different than the usual A*B multiplication
	// We copy blocks from the required columns directly as squares and use double buffering

	// Also note that A is already in Column-Major (from cpu) to reduce bank conflicts

	__shared__ double sm_col_onY[2][TILE_SIZE][TILE_SIZE];
	__shared__ double sm_col_onX[2][TILE_SIZE][TILE_SIZE];

	double result = 0.0;


	// Column to copy for this thread based on the C tile's x
	const int col_x = block_start_x + tx;
	
	// Column to copy for this thread based on the C tile's y
	const int col_y = block_start_y + tx;

	int m = 0;


	sm_col_onX[0][ty][tx] = src[AT(m + ty, col_x, nr_cols)];
	sm_col_onY[0][ty][tx] = src[AT(m + ty, col_y, nr_cols)];

	for (; m < nr_rows;) {
		__syncthreads();

		m += TILE_SIZE;

		sm_col_onX[1][ty][tx] = src[AT(m + ty, col_x, nr_cols)];
		sm_col_onY[1][ty][tx] = src[AT(m + ty, col_y, nr_cols)];

		#pragma unroll
		for (int k = 0; k < TILE_SIZE; ++k) {
			result += sm_col_onX[0][k][tx] * sm_col_onY[0][k][ty];
		}


		if (m >= nr_rows) {
			break;
		}
		__syncthreads();

		m += TILE_SIZE;


		sm_col_onX[0][ty][tx] = src[AT(m + ty, col_x, nr_cols)];
		sm_col_onY[0][ty][tx] = src[AT(m + ty, col_y, nr_cols)];

		#pragma unroll
		for (int k = 0; k < TILE_SIZE; ++k) {
			result += sm_col_onX[1][k][tx] * sm_col_onY[1][k][ty];
		}

	}

	output[AT(res_row, res_col, nr_cols)] = result;
	output[AT(res_col, res_row, nr_cols)] = result;
}

__global__ void opt_dev_Tmultiply(int nr_rows, int nr_cols, double* src, double* result)
{
	if (blockIdx.x < blockIdx.y) {
		return;
	}

	const int row = blockIdx.x * TILE_SIZE + threadIdx.x;
	const int col = blockIdx.y * TILE_SIZE + threadIdx.y;


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
		result[AT(row, col, nr_cols)] = blockIdx.x + blockIdx.y * 100;
	}
}



constexpr int T_GRANUL = 8;
constexpr int WORK_THREAD = TILE_SIZE / T_GRANUL;

__global__ void opt_dev_TmultiplyOdd(int nr_rows, int nr_cols, double* src, double* output);
__global__ void opt_dev_Tmultiply_RegCache(int nr_rows, int nr_cols, double* src, double* output);

Matrix opt_Tmutliply(Matrix& input) {
	Matrix result(input.cols, input.cols);
	result.AllocDevice();



	input.IntoDevMatrix();

	constexpr int Threads = BLOCK_SIZE;
	int GridSize = div_ceil(input.cols, TILE_SIZE);

	dim3 block(Threads, Threads);
	dim3 grid(GridSize, GridSize);

	

	optimisedTime = CountTime([&]() {
		opt_dev_Tmultiply <<<grid, block >> > (input.rows, input.cols, input.dev_data, result.dev_data);
	}); 

	printf("Double Buffer: %f\n", optimisedTime);


	optimisedTime = CountTime([&]() {
		opt_dev_Tmultiply_RegCache << <grid, block >> > (input.rows, input.cols, input.dev_data, result.dev_data);
	});
	
	printf("RegCache: %f\n", optimisedTime);


	//GridSize = div_ceil(input.cols, TILE_SIZE);
	grid = dim3(GridSize, GridSize / T_GRANUL);


	optimisedTime = CountTime([&]() {
			opt_dev_TmultiplyOdd<< <grid, block>>> (input.rows, input.cols, input.dev_data, result.dev_data);
		});
	
	printf("Granularity: %f\n", optimisedTime);


	result.FromDevMatrix();

	input.FreeDevice();
	result.FreeDevice();

	return result;
}

constexpr int RC_WORK = TILE_SIZE;


__global__ void opt_dev_Tmultiply_RegCache(int nr_rows, int nr_cols, double* src, double* output)
{
	const int bx = blockIdx.x;
	const int by = blockIdx.y;
	
	if (by > bx) {
		return;
	}

	const int row = bx * TILE_SIZE + threadIdx.x;
	const int col = by * TILE_SIZE + threadIdx.y;

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	const int block_start_x = bx * TILE_SIZE;
	const int block_start_y = by * TILE_SIZE;


	const int cl_ty = threadIdx.y;

	// 
	__shared__ double sm_col_onY[RC_WORK][TILE_SIZE];
	__shared__ double sm_col_onX[RC_WORK][TILE_SIZE];

	// Column to copy for this thread based on the C tile's x
	const int col_x = block_start_x + tx;

	// Column to copy for this thread based on the C tile's y
	const int col_y = block_start_y + tx;

	

	double result = 0.0;
	double reg_cache_X;
	double reg_cache_Y;


	int m = 0;

	reg_cache_X = src[AT(m + cl_ty, col_x, nr_rows)];
	reg_cache_Y = src[AT(m + cl_ty, col_y, nr_rows)];

	
	for (m = RC_WORK; m < nr_rows; m += RC_WORK) {
		sm_col_onX[ty][tx] = reg_cache_X;
		sm_col_onY[ty][tx] = reg_cache_Y;

		__syncthreads();

		reg_cache_X = src[AT(m + ty, col_x, nr_cols)];
		reg_cache_Y = src[AT(m + ty, col_y, nr_cols)];

		#pragma unroll
		for (int k = 0; k < RC_WORK; ++k) {
			result += sm_col_onX[k][tx] * sm_col_onY[k][ty];
		}
	} 

	for (int k = 0; k < RC_WORK; ++k) {
		sm_col_onX[ty][tx] = reg_cache_X;
		sm_col_onY[ty][tx] = reg_cache_Y;

		__syncthreads();
		
		#pragma unroll
		for (int k = 0; k < RC_WORK; ++k) {
			result += sm_col_onX[k][tx] * sm_col_onY[k][ty];
		}
	}


	output[AT(row, col, nr_cols)] = result;
	output[AT(col, row, nr_cols)] = result;
	


	constexpr DebugOutput db2 = DebugOutput::Result;

	if (db2 == DebugOutput::Result) {

	}
	else if (db2 == DebugOutput::RowCol) {
		output[AT(row, col, nr_cols)] = row + col * 100;
	}
	else if (db2 == DebugOutput::Thread) {
		output[AT(row, col, nr_cols)] = threadIdx.x + threadIdx.y * 100;
	}
	else if (db2 == DebugOutput::Block) {
		output[AT(row, col, nr_cols)] = blockIdx.x + blockIdx.y * 100;
	}
}







// Using different techniques of optimization for Odd columns
// enables us to disable double buffering for this case and have more registers to work with, 
// Implements other methods that we tried but found suboptimal



__global__ void opt_dev_TmultiplyOdd(int nr_rows, int nr_cols, double* src, double* output)
{
	const int bx = blockIdx.x;
	const int by = blockIdx.y * T_GRANUL;
	
	if (by > bx) {
		return;
	}

	const int row = bx * TILE_SIZE + threadIdx.x;
	const int col = by * TILE_SIZE + threadIdx.y;

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;



	const int block_start_x = bx * TILE_SIZE;
	const int block_start_y = by * TILE_SIZE;

	const int threadCopyOffset = ty / WORK_THREAD;

	const int cl_ty = threadIdx.y % WORK_THREAD;

	// 
	__shared__ double sm_col_onY[T_GRANUL][WORK_THREAD][TILE_SIZE];
	__shared__ double sm_col_onX[WORK_THREAD][TILE_SIZE];

	// Column to copy for this thread based on the C tile's x
	const int col_x = block_start_x + tx;

	// Column to copy for this thread based on the C tile's y
	const int col_y = block_start_y + tx;

	int m = 0; 


	double result[T_GRANUL];
	//double reg_cache_X;
	//double reg_cache_Y;


	#pragma unroll
	for (int i = 0; i < T_GRANUL; ++i) {
		result[i] = 0.0;
	}

	
	for (m = 0; m < nr_rows; m += WORK_THREAD) {

		sm_col_onX[ty % WORK_THREAD][tx]                   = src[AT(m + cl_ty, col_x, nr_cols)];;
		sm_col_onY[threadCopyOffset][ty % WORK_THREAD][tx] = src[AT(m + cl_ty, col_y + (threadCopyOffset * TILE_SIZE), nr_cols)];;

		__syncthreads();

		#pragma unroll
		for (int i = 0; i < T_GRANUL; ++i) {
			#pragma unroll
			for (int k = 0; k < WORK_THREAD; ++k) {
				result[i] += sm_col_onX[k][tx] * sm_col_onY[i][k][ty];
			}
		}
	} 


	#pragma unroll
	for (int i = 0; i < T_GRANUL; ++i) {
		int out_col = col + i * TILE_SIZE;
		output[AT(row, out_col, nr_cols)] = result[i];
		output[AT(out_col, row, nr_cols)] = result[i];
		
		//output[AT(row, out_col, nr_cols)] = (blockIdx.x * 10 + blockIdx.y * 100 + i);
		//output[AT(out_col, row, nr_cols)] = (blockIdx.x + blockIdx.y * 100 + i * 10000) + 1;
	}



	constexpr DebugOutput db2 = DebugOutput::Result;

	if (db2 == DebugOutput::Result) {

	}
	else if (db2 == DebugOutput::RowCol) {
		output[AT(row, col, nr_cols)] = row + col * 100;
	}
	else if (db2 == DebugOutput::Thread) {
		output[AT(row, col, nr_cols)] = threadIdx.x + threadIdx.y * 100;
	}
	else if (db2 == DebugOutput::Block) {
		output[AT(row, col, nr_cols)] = blockIdx.x + blockIdx.y * 100;
	}
}
