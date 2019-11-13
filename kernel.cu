#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_error_macros.h"
#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <initializer_list>

static __inline__ Matrix cublas_Tmultiply(Matrix& inputMatrix) {
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

int main() {
	Matrix inputMatrix(4, {
		0,  1,  2,  3,
		10, 11, 12, 13,
		20, 21, 22, 23
	});

	cudaSetDevice(0); CE;

	Matrix resultMatrix = cublas_Tmultiply(inputMatrix);

	printf("Result: \n");
	resultMatrix.Print();
	return 0;
}
