#include "Lesson2.h"
#include <cudaDefs.h>

namespace lesson2 {
	//const size_t N = 10;
	const size_t Rows = 10;
	const size_t Cols = 5;
	const size_t BlockSize = 2;

	__global__ void fill(int* matrix, size_t rows, size_t cols, size_t pitch)
	{
		int row = blockIdx.x * BlockSize + threadIdx.x;
		int col = blockIdx.y * BlockSize + threadIdx.y;
		if (row >= rows || col >= cols)
			return;

		int index = col * pitch + row;
		int value = col * rows + row;
		printf("r=%-5d c=%-5d index=%-5d v=%-5d\n", col, row, index, value);
		matrix[index] = value;
	}

	__global__ void increment(int* matrix, size_t rows, size_t cols, size_t pitch)
	{
		int row = blockIdx.x * BlockSize + threadIdx.x;
		int col = blockIdx.y * BlockSize + threadIdx.y;
		if (row >= rows || col >= cols)
			return;

		int index = col * pitch + row;
		int value = col * rows + row;
		matrix[index]++;
	}

	void run()
	{
		int *dMatrix;
		size_t pitchInBytes = 0;
		size_t rowsInBytes = Cols * sizeof(int);
		checkCudaErrors(cudaMallocPitch((void**)&dMatrix, &pitchInBytes, rowsInBytes, Rows));
		size_t pitch = pitchInBytes / sizeof(int);
		dim3 grid = dim3(getNumberOfParts(Rows, BlockSize), getNumberOfParts(Cols, BlockSize));
		dim3 block = dim3(BlockSize, BlockSize);

		fill << <grid, block >> > (dMatrix, Rows, Cols, pitch);
		checkDeviceMatrix(dMatrix, pitchInBytes, Cols, Rows, "%-3d ", "dMatrix");

		increment << <grid, block >> > (dMatrix, Rows, Cols, pitch);
		checkDeviceMatrix(dMatrix, pitchInBytes, Cols, Rows, "%-3d ", "dMatrix");

		int *matrix = new int[Rows * Cols];
		checkCudaErrors(cudaMemcpy2D(matrix, Rows * sizeof(int), dMatrix, pitchInBytes, Rows * sizeof(int), Cols, cudaMemcpyKind::cudaMemcpyDeviceToHost));
		checkHostMatrix(matrix, Rows * sizeof(int), Cols, Rows, "%-3d ", "matrix");
	}
}