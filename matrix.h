#pragma once
#include "cuda_error_macros.h"
#include <memory>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

// Represents manages a Matrix and its memory on Host and Device.
// Automatically frees both Host & Device memory when going out of scope ensuring no memory leaks.
struct Matrix {
	// NOTE: Data is stored in row major. dev_data is dependant on the function IntoDevMatrix used.
	float* data{ nullptr };
	float* dev_data{ nullptr };

	int rows{ 0 };
	int cols{ 0 };

	Matrix() {}

	Matrix(int inCols, int inRows) {
		cols = inCols;
		rows = inRows;

		AllocHost();
	}

	// Expects row major order in inData
	Matrix(int inCols, std::initializer_list<float> inData) {
		size_t size = inData.size();

		cols = inCols;
		rows = (int)size / cols;

		AllocHost();

		float* dataPtr = data;

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

	// Move assignment
	Matrix& operator=(Matrix&& other) {
		FreeHost();
		FreeDevice();

		data = other.data;
		dev_data = other.dev_data;
		cols = other.cols;
		rows = other.rows;

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
		data = (float*)malloc(Size() * sizeof(float));
	}

	// Frees and then (re)allocates new Device memory.
	// Note: GPU data is lost.
	void AllocDevice() {
		FreeDevice();
		cudaMalloc((void**)&dev_data, Size() * sizeof(float)) CE;
	}


	//
	// Host <-> Device transfers
	//

	// Copies the Host buffer to the device buffer. Automatically allocates Device buffer if required.
	void IntoDevMatrix() {
		if (!dev_data) {
			AllocDevice();
		}
		cudaMemcpy(dev_data, (const void*)data, Size() * sizeof(float), cudaMemcpyHostToDevice); CE;
	}

	// Copies a transposed version of the host buffer to the device buffer. Automatically allocates Device buffer if required.
	void IntoDevMatrix_ColMajor() {
		if (!dev_data) {
			AllocDevice();
		}

		float* t = Transposed();
		cudaMemcpy(dev_data, (const void*)t, Size() * sizeof(float), cudaMemcpyHostToDevice); CE;
		free(t);
	}


	// Copies the Device buffer back to host. Automatically allocates Host buffer if required.
	void FromDevMatrix() {
		if (!data) {
			AllocHost();
		}
		cudaMemcpy(data, dev_data, Size() * sizeof(float), cudaMemcpyDeviceToHost); CE;
	}

	// Copies a transposed version of the Device buffer back to host. Automatically allocates Host buffer if required.
	void FromDevMatrix_ColMajor() {
		if (!data) {
			AllocHost();
		}
		cudaMemcpy(data, dev_data, Size() * sizeof(float), cudaMemcpyDeviceToHost); CE;
		// Transpose and swap.
		float* fixed = Transposed();
		free(data);
		data = fixed;
	}


	//
	// Misc utilities
	//

	int Size() {
		return rows * cols;
	}

	float At(int i, int j) {
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
	float* Transposed() {
		float* t = (float*)malloc(Size() * sizeof(float));

		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				t[IDX2C(i, j, rows)] = At(i, j);
			}
		}
		return t;
	}
};
