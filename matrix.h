#pragma once
#include "cuda_error_macros.h"
#include <memory>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

// Represents manages a Matrix and its memory on Host and Device.
// Automatically frees both Host & Device memory when going out of scope ensuring no memory leaks.
struct Matrix {
	// NOTE: Data is stored in row major. dev_data is dependant on the function IntoDevMatrix used.
	double* data{ nullptr };
	double* dev_data{ nullptr };

	int rows{ 0 };
	int cols{ 0 };

	Matrix() {}

	Matrix(int inCols, int inRows) {
		cols = inCols;
		rows = inRows;
	}

	// Expects row major order in inData
	Matrix(int inCols, std::initializer_list<double> inData) {
		size_t size = inData.size();

		cols = inCols;
		rows = (int)size / cols;

		AllocHost();

		double* dataPtr = data;

		for (auto f : inData) {
			*(dataPtr++) = f;
		}
	}

	~Matrix() {
		FreeHost();
		FreeDevice();
	}

	//
	// Move & Copy Operators
	//

	// Move assignment, transfer the owned memory to the other matrix without copy
	Matrix& operator=(Matrix&& other) {
		// Cleanup whatever is currently allocated
		FreeHost();
		FreeDevice();

		// Copy pointers & data
		data = other.data;
		dev_data = other.dev_data;
		cols = other.cols;
		rows = other.rows;

		// Remember to modify these to prevent free from 'other' matrix
		other.data = nullptr;
		other.dev_data = nullptr;
		return *this;
	}

	// Move constructor for Matrix
	Matrix(Matrix&& other) {
		*this = std::move(other);
	}

	// Copy operators should actually just memcpy, but delete them so we get compile errors if we copy by accident
	Matrix(const Matrix&) = delete;
	Matrix& operator=(const Matrix&) = delete;


	//
	// Host & Device Memory
	//

	// Frees CPU memory IF allocated
	void FreeHost() {
		if (data != nullptr) {
			free(data);
			data = nullptr;
		}
	}

	// Frees GPU memory IF allocated
	void FreeDevice() {
		if (dev_data != nullptr) {
			cudaFree(dev_data);
			dev_data = nullptr;
		}
	}

	// Frees and then (re)allocates new Host memory.
	// Note: CPU data is lost.
	void AllocHost() {
		FreeHost();
		data = (double*)malloc(Size() * sizeof(double));
	}

	// Frees and then (re)allocates new Device memory.
	// Note: GPU data is lost.
	void AllocDevice() {
		FreeDevice();
		cudaMalloc((void**)&dev_data, Size() * sizeof(double)) CE;
	}


	//
	// Host <-> Device transfers
	//

	// Copies the Host buffer to the device buffer. Automatically allocates Device buffer if required.
	void IntoDevMatrix() {
		if (!dev_data) {
			AllocDevice();
		}
		cudaMemcpy(dev_data, (const void*)data, Size() * sizeof(double), cudaMemcpyHostToDevice); CE;
	}

	// Copies a transposed version of the host buffer to the device buffer. Automatically allocates Device buffer if required.
	void IntoDevMatrix_ColMajor() {
		if (!dev_data) {
			AllocDevice();
		}

		double* t = Transposed();
		cudaMemcpy(dev_data, (const void*)t, Size() * sizeof(double), cudaMemcpyHostToDevice); CE;
		free(t);
	}


	// Copies the Device buffer back to host. Automatically allocates Host buffer if required.
	void FromDevMatrix() {
		if (!data) {
			AllocHost();
		}
		cudaMemcpy(data, dev_data, Size() * sizeof(double), cudaMemcpyDeviceToHost); CE;
	}

	// Copies a transposed version of the Device buffer back to host. Automatically allocates Host buffer if required.
	void FromDevMatrix_ColMajor() {
		if (!data) {
			AllocHost();
		}
		cudaMemcpy(data, dev_data, Size() * sizeof(double), cudaMemcpyDeviceToHost); CE;
		// Transpose and swap.
		double* fixed = Transposed();
		free(data);
		data = fixed;
	}


	//
	// Misc utilities
	//
	double** Dev_As2D() {
		return (double**)dev_data;
	}

	int Size() {
		return rows * cols;
	}

	double At(int i, int j) {
		return data[i * cols + j];
	}

	void Print() {
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				printf("%7.0f", At(i, j));
			}
			printf("\n");
		}
	}

	// mallocs the temporary returned buffer, you should free after use
	double* Transposed() {
		double* t = (double*)malloc(Size() * sizeof(double));

		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				t[IDX2C(i, j, rows)] = At(i, j);
			}
		}
		return t;
	}

	// Compares Host Data with "other's" data. Costs O(N) when equal because we need to apply delta to each value
	bool IsNearlyEqual(const Matrix& other, double delta = 1e-3) {
		if (cols != other.cols || rows != other.rows) {
			return false;
		}

		for (int i = 0; i < Size(); ++i) {
			if (abs(data[i] - other.data[i]) > delta) {
				return false;
			}
		}
		return true;
	}
};
