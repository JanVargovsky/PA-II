#include "Lesson2.h"
#include <cudaDefs.h>

namespace lesson2 {
	const size_t N = 10;
	const size_t Rows = N;
	const size_t Cols = N;
	const size_t BlockSize = 8;

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
		size_t rowsInBytes = N * sizeof(int);
		size_t cols = N;
		checkCudaErrors(cudaMallocPitch((void**)&dMatrix, &pitchInBytes, rowsInBytes, cols));
		size_t pitch = pitchInBytes / sizeof(int);
		dim3 grid = dim3(getNumberOfParts(N, BlockSize), getNumberOfParts(N, BlockSize));
		dim3 block = dim3(BlockSize, BlockSize);

		fill << <grid, block >> > (dMatrix, N, N, pitch);
		checkDeviceMatrix(dMatrix, pitchInBytes, N, N, "%-3d ", "dMatrix");

		increment << <grid, block >> > (dMatrix, N, N, pitch);
		checkDeviceMatrix(dMatrix, pitchInBytes, N, N, "%-3d ", "dMatrix");
	}
}