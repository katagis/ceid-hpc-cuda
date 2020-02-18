#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"

#include <stdio.h>
#include <stdlib.h>
#include <initializer_list>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>

// Contains our "Matrix" class that represents a unified matrix with both cpu and gpu data
// along with utility functions for memory allocations and host <-> device transfers.
// This class automatically deallocates all its host and device resources when going out of scope.
#include "matrix.h"


// Resolves a row major position based on Row and Column
#define AT(row, col, nr_cols) ((row) * (nr_cols) + (col))

// Timers for the results.
float cublasTime = 0.f;
float basicTime = 0.f;
float optimisedTime = 0.f;

// Utility enum that was used for debugging during development
enum class DebugOutput {
	Result = 0,
	RowCol = 1,
	Thread = 2,
	Block = 3
};
constexpr DebugOutput debugOutput = DebugOutput::Result;

// Debug flag used during development to verify correctness of the kernels
constexpr bool checkResultsForCorrectness = false;

// Also used as block size
constexpr int TILE_SIZE = 32;

//
// General Utility Functions
//

// Generates the input matrix. (Done on cpu)
Matrix GetInputMatrix(int TestRows = 5000, int TestCols = 5000) {

	Matrix inputMatrix(TestCols, TestRows);
	inputMatrix.AllocHost();

	for (int i = 0; i < inputMatrix.Size(); ++i) {
		inputMatrix.data[i] = std::rand() / 100.f;
	}

	for (int i = 0; i < TestCols; ++i) {
		inputMatrix.data[i] = 0 * 10;
	}

	return std::move(inputMatrix);
}

// Returns milliseconds for the time Lambda f took to run on gpu.
template<typename Lambda>
float CountTime(Lambda&& f) {
	
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


//
// Main
//

// Forward declare the kernel launcher functions
Matrix cublas_Tmultiply(Matrix& input);
Matrix opt_Tmutliply(Matrix& input);
Matrix basic_Tmutliply(Matrix& input);

int main() {
	std::vector<std::pair<int, int>> TestSizes;
	
	// NOTE: The original tests may use up to 10 gb of host memory.
	// For easier running and stability the test sizes have been reduced in the source code 
	// after gathering execution data

	for (int i = 1024; i <= 8500; i = i += 1024)	{
		TestSizes.push_back({ i, i });
	}

	for (int i = 1000; i <= 9000; i += 1000) {
		TestSizes.push_back({ div_ceil(i, 3), i });
	}

	std::right(std::cout);

	std::cout << 
		std::setw(7) << "Rows" <<
		std::setw(7) << "Cols" << 
		std::setw(10) << "cublas" << 
		std::setw(10) << "basic" <<
		std::setw(10) << "optim" << std::endl;

	cudaSetDevice(0); CE;

	for (auto& test : TestSizes) {
		Matrix inputMatrix = GetInputMatrix(test.first, test.second);

		Matrix cublasResult = cublas_Tmultiply(inputMatrix);
		if (!checkResultsForCorrectness) {
			// We won't need the cpu data. Save some host memory
			cublasResult.FreeHost();
		}

		Matrix basicCudaResult = basic_Tmutliply(inputMatrix);
		if (!checkResultsForCorrectness) {
			basicCudaResult.FreeHost();
		}

		Matrix optCudaResult = opt_Tmutliply(inputMatrix);
		if (!checkResultsForCorrectness) {
			optCudaResult.FreeHost();
		}

		if (checkResultsForCorrectness && !cublasResult.IsDeltaEqual(optCudaResult)) {
			printf("Optimised cuda had different results:\n");
			printf("cublas:\n");
			cublasResult.Print();

			printf("cuda:\n");
			optCudaResult.Print();
		}
		std::cout <<
			std::setw(7) << test.first <<
			std::setw(7) << test.second <<
			std::setw(10) << cublasTime <<
			std::setw(10) << basicTime <<
			std::setw(10) << optimisedTime << std::endl;
	}
	

	cudaDeviceReset();

	// Our matrix memory will get freed through the custom destructor if not already freed.
	return 0;
}

//
// cublas Implementation
//
Matrix cublas_Tmultiply(Matrix& inputMatrix) {
	const double alpha = 1;
	const double beta = 0;

	Matrix result(inputMatrix.cols, inputMatrix.cols);
	result.AllocDevice(); // Allocates device memory for this matrix using cudaMalloc

	inputMatrix.IntoDevMatrix_ColMajor(); // Uploads the buffer to the device memory directly in Column Major using cudaMemcpy

	
	cublasHandle_t handle;
	auto status = cublasCreate(&handle); CBE(status);

	int N = inputMatrix.cols;
	int M = inputMatrix.rows;

	// Warmup for cublas.
	status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, M, &alpha,
						 inputMatrix.dev_data, M, inputMatrix.dev_data, M, &beta, result.dev_data, N);
	CBE(status);


	cublasTime = CountTime([&]() {
		status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, M, &alpha,
							 inputMatrix.dev_data, M, inputMatrix.dev_data, M, &beta, result.dev_data, N);
	});
	CBE(status);
	cublasDestroy(handle);


	result.FromDevMatrix_ColMajor(); // Retrieves the buffer from device memory in cpu memroy of the matrix using cudaMemcpy

	result.FreeDevice(); // cudaFrees result's device memroy
	inputMatrix.FreeDevice();// cudaFrees the result device memroy

	return std::move(result);
}


//
// Kernel
// Basic 'naive' multiplication 
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

// Util kernel launcher for naive implementation
Matrix basic_Tmutliply(Matrix& input) {
	Matrix result(input.cols, input.cols);
	result.AllocDevice();

	input.IntoDevMatrix();

	constexpr int Threads = TILE_SIZE;

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



//
// OPTIMISED CUDA
//

//
// Implementation Notes:
// 1. T(A) * A is symmetrical
// 2. Tile based
// 3. Double buffering
// 4. Supports matrix sizes that misalign with the block size
//

//
// Notes about Thread Granularity (not implemented): 
// We tried multiple thread granularity implementations: 1x2, 1x4, 2x2, 3x3.
// In our tests thread granularity performed better than most other optmisations.
// The main limitation of granularity was the resources required (shared memory & registers). 
// Due to that, we could not combine thread granularity with any memory prefetch methods (double buffering / register prefetch) without running out of thread resources.
// In the end the simple double buffering was just faster than just granularity without memory prefetch.
//

__device__ void opt_dev_Tmultiply_Impl(int nr_rows, int nr_cols, double* src, double* output, const int res_row, const int res_col);

//
// For TILE_SIZE = 32 this kernel uses:
// ptxas: 62 registers, 32768 bytes smem, 376 bytes cmem[0]
// 
// which matches extactly what our device supports for max occupancy:
// 64 registers and 32768 bytes shared mem per thread
//

// Basic handler for the actual function. Handles the trianglular array and debugging
__global__ void opt_dev_Tmultiply(int nr_rows, int nr_cols, double* src, double* result)
{
	// Handles Symmetrical triangle
	if (blockIdx.x < blockIdx.y) {
		return;
	}

	const int row = blockIdx.x * TILE_SIZE + threadIdx.x;
	const int col = blockIdx.y * TILE_SIZE + threadIdx.y;

	opt_dev_Tmultiply_Impl(nr_rows, nr_cols, src, result, row, col);

	// Debug output. Constexpr enables the compiler to always remove this at compile time.
	if (debugOutput != DebugOutput::Result) {
		if (row >= nr_cols || col >= nr_cols) {
			return;
		}

		if (debugOutput == DebugOutput::RowCol) {
			result[AT(row, col, nr_cols)] = row + col * 100;
		}
		else if (debugOutput == DebugOutput::Thread) {
			result[AT(row, col, nr_cols)] = threadIdx.x + threadIdx.y * 100;
		}
		else if (debugOutput == DebugOutput::Block) {
			result[AT(row, col, nr_cols)] = blockIdx.x + blockIdx.y * 100;
		}
	}
}

// The actual algorithm kernel
__device__ void opt_dev_Tmultiply_Impl(int nr_rows, int nr_cols, double* src, double* output, const int res_row, const int res_col)
{
	// transpose(A) * A in practice means we multiply between 2 columns of the src.
	// The terms column, column_X and column_Y are used meaning the columns from the "src" that are needed to be multiplied.

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	const int bx = blockIdx.x;
	const int by = blockIdx.y;


	// The first X coordinate index of this block on the output matrix
	const int block_start_x = bx * TILE_SIZE;
	
	// The first Y coordinate index of this block on the output matrix
	const int block_start_y = by * TILE_SIZE;

	
	__shared__ double sm_col_onY[2][TILE_SIZE][TILE_SIZE]; // Double buffer based on the result block Y coordinate
	__shared__ double sm_col_onX[2][TILE_SIZE][TILE_SIZE]; // Double buffer based on the result block X coordinate

	double result = 0.0; // Accumulates the result for this output tile.


	// Column to copy for this thread based on the C tile's x
	const int col_x = block_start_x + tx;

	// Column to copy for this thread based on the C tile's y
	const int col_y = block_start_y + tx;
	// Note: we copy based on threadX for both columns, this improves memory access paterns
	// tremendously and improves the running time by up to 20%


	// Stores overflow states on each dimension and in general for this particlar thread
	const int overflows_x = block_start_x + tx >= nr_cols; 
	const int overflows_y = block_start_y + tx >= nr_cols;
	const int overflows = res_row >= nr_cols || res_col >= nr_cols;


	int m = 0;

	// Prepare the first buffer
	sm_col_onX[0][ty][tx] = src[AT(0 + ty, col_x, nr_cols)];
	sm_col_onY[0][ty][tx] = src[AT(0 + ty, col_y, nr_cols)];

	for (;;) {
		__syncthreads();

		// Prefetch a TILE on buffer[1], multiply based on buffer[0]
		m += TILE_SIZE;

		// If the next tile exists prefetch it, otherwise just calculate the current one.
		if (m + TILE_SIZE < nr_rows) {

			// Begin prefetching next tile
			if (!overflows_x) { // Using this if check is faster than introducing bank conflicts from clamping to the edges.
				sm_col_onX[1][ty][tx] = src[AT(m + ty, col_x, nr_cols)];
			}

			if (!overflows_y) {
				sm_col_onY[1][ty][tx] = src[AT(m + ty, col_y, nr_cols)];
			}

			// Calculate based on the previous tile, note that no __syncthreads is required here due to double buffering
			#pragma unroll
			for (int k = 0; k < TILE_SIZE; ++k) {
				result += sm_col_onX[0][k][tx] * sm_col_onY[0][k][ty];
			}
		}
		else {
			#pragma unroll
			for (int k = 0; k < TILE_SIZE; ++k) {
				result += sm_col_onX[0][k][tx] * sm_col_onY[0][k][ty];
			}

			break;
		}


		// Prefetch a TILE on buffer[0], multiply based on buffer[1]
		// Next tile, exactly same code but buffers are swapped
		// (code duplication was required here to gain full optimisation benefits, the compiler failed to optimise the odd-even loop)
		__syncthreads();
		m += TILE_SIZE;

		if (m + TILE_SIZE < nr_rows) {
			if (!overflows_x) {
				sm_col_onX[0][ty][tx] = src[AT(m + ty, col_x, nr_cols)];
			}

			if (!overflows_y) {
				sm_col_onY[0][ty][tx] = src[AT(m + ty, col_y, nr_cols)];
			}

			#pragma unroll
			for (int k = 0; k < TILE_SIZE; ++k) {
				result += sm_col_onX[1][k][tx] * sm_col_onY[1][k][ty];
			}
		}
		else {
			#pragma unroll
			for (int k = 0; k < TILE_SIZE; ++k) {
				result += sm_col_onX[1][k][tx] * sm_col_onY[1][k][ty];
			}

			break;
		}
	}

	if (!overflows) {
		// Calculate the remaining odd elements
		for (int i = m; i < nr_rows; ++i) {
			result += src[AT(i, res_row, nr_cols)] * src[AT(i, res_col, nr_cols)];
		}


		output[AT(res_row, res_col, nr_cols)] = result;
		output[AT(res_col, res_row, nr_cols)] = result; // write symmetrical results
	}
}


// Launches the optimised multiplication kernel
Matrix opt_Tmutliply(Matrix& input) {
	Matrix result(input.cols, input.cols);
	result.AllocDevice();

	input.IntoDevMatrix();

	constexpr int Threads = TILE_SIZE;
	int GridSize = div_ceil(input.cols, Threads);

	dim3 block(Threads, Threads);
	dim3 grid(GridSize, GridSize);

	optimisedTime = CountTime([&]() {
		opt_dev_Tmultiply <<<grid, block >> > (input.rows, input.cols, input.dev_data, result.dev_data);
	}); 

	result.FromDevMatrix();

	input.FreeDevice();
	result.FreeDevice();

	return result;
}
