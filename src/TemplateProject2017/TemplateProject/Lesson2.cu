#include "Lesson2.h"
#include <cudaDefs.h>
#include <cassert>

namespace lesson2 {
	const size_t Rows = 15;
	const size_t Cols = 20;
	const size_t BlockSize = 3;

	__global__ void fill(int* matrix, size_t rows, size_t cols, size_t pitch)
	{
		int row = blockIdx.x * BlockSize + threadIdx.x;
		int col = blockIdx.y * BlockSize + threadIdx.y;
		if (row >= rows || col >= cols)
			return;

		int index = row * pitch + col;
		int value = col * rows + row;
		//printf("r=%-5d c=%-5d index=%-5d v=%-5d\n", row, col, index, value);
		matrix[index] = value;
	}

	__global__ void increment(int* matrix, size_t rows, size_t cols, size_t pitch)
	{
		int row = blockIdx.x * BlockSize + threadIdx.x;
		int col = blockIdx.y * BlockSize + threadIdx.y;
		if (row >= rows || col >= cols)
			return;

		int index = row * pitch + col;
		int value = col * rows + row;
		matrix[index]++;
	}

	template<typename T>
	bool arraysEqual(T *a, T *b, size_t length)
	{
		for (size_t i = 0; i < length; i++)
			if (a[i] != b[i])
				return false;
		return true;
	}

	void run()
	{
		int *dMatrix;
		size_t pitchInBytes = 0;
		checkCudaErrors(cudaMallocPitch((void**)&dMatrix, &pitchInBytes, Cols * sizeof(int), Rows));
		size_t pitch = pitchInBytes / sizeof(int);
		dim3 grid = dim3(getNumberOfParts(Rows, BlockSize), getNumberOfParts(Cols, BlockSize));
		dim3 block = dim3(BlockSize, BlockSize);

		fill << <grid, block >> > (dMatrix, Rows, Cols, pitch);
		checkDeviceMatrix(dMatrix, pitchInBytes, Rows, Cols, "%-3d ", "dMatrix");

		increment << <grid, block >> > (dMatrix, Rows, Cols, pitch);
		checkDeviceMatrix(dMatrix, pitchInBytes, Rows, Cols, "%-3d ", "dMatrix");

		int *expectedMatrix = new int[Rows * Cols];
		for (size_t i = 0; i < Rows * Cols; i++)
			expectedMatrix[i] = i + 1;

		int *matrix = new int[pitch * Rows];
		checkCudaErrors(cudaMemcpy2D(matrix, pitchInBytes, dMatrix, pitchInBytes, Cols * sizeof(int), Rows, cudaMemcpyKind::cudaMemcpyDeviceToHost));
		checkHostMatrix(matrix, pitchInBytes, Rows, Cols, "%-3d ", "matrix");

		//assert(arraysEqual(expectedMatrix, matrix, Rows * Cols));
		
		delete[] matrix;
		delete[] expectedMatrix;
		cudaFree(dMatrix);
	}
}