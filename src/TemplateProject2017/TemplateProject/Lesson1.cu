#include "Lesson1.h"
#include <cudaDefs.h>
#include <string>
#include <iostream>
#include <cassert>

namespace lesson1 {

	using namespace std;

	const int BlockSize = 512;
	const int ThreadsPerBlock = 8;

	const int DebugPrintArrayCount = 20;

	template<typename T>
	T* createHostArray(size_t size)
	{
		return new T[size];
	}

	template<typename T>
	T* createDeviceArray(size_t sizeInBytes)
	{
		T* array = nullptr;
		checkCudaErrors(cudaMalloc((void**)&array, sizeInBytes));
		return array;
	}

	template<typename T>
	void fill(T* a, size_t size, int increment = 1)
	{
		int value = 0;
		for (int i = 0; i < size; i++, value += increment)
			a[i] = value;
	}

	template<typename T>
	bool arraysEqual(T *a, T *b, size_t length)
	{
		for (size_t i = 0; i < length; i++)
			if (a[i] != b[i])
				return false;
		return true;
	}

	void runAdd1(int *a, int *b, int *c, size_t length, size_t lengthInBytes);
	void runAdd2(int *a, int *b, int *c, size_t length, size_t lengthInBytes);

	void run()
	{
		srand(42);
		const size_t N = 20000000;
		const size_t NB = N * sizeof(int);

		auto expected = createHostArray<int>(N);
		fill(expected, N, 2);

		auto a = createHostArray<int>(N);
		fill(a, N);
		checkHostMatrix<int>(a, 1, 1, DebugPrintArrayCount, "%d ", "a");

		auto b = createHostArray<int>(N);
		fill(b, N);
		checkHostMatrix<int>(b, 1, 1, DebugPrintArrayCount, "%d ", "b");
		auto c = createHostArray<int>(N);

		cout << "runAdd1" << endl;
		runAdd1(a, b, c, N, NB);
		assert(arraysEqual(expected, c, N));
		checkHostMatrix<int>(c, 1, 1, DebugPrintArrayCount, "%d ", "c");

		cout << "runAdd2" << endl;
		runAdd2(a, b, c, N, NB);
		assert(arraysEqual(expected, c, N));
		checkHostMatrix<int>(c, 1, 1, DebugPrintArrayCount, "%d ", "c");

		delete[] a;
		delete[] b;
		delete[] c;
	}

	template<typename T>
	__global__ void addClassic(T *a, T *b, T *c, size_t length)
	{
		auto i = blockDim.x * blockIdx.x + threadIdx.x;
		//printf("%d %d %d, i=%d\n", blockDim.x, blockIdx.x, threadIdx.x, i);
		//printf("i=%d\n", i);
		if (i >= length)
			return;
		c[i] = a[i] + b[i];
	}

	void runAdd1(int *a, int *b, int *c, size_t length, size_t lengthInBytes)
	{
		auto da = createDeviceArray<int>(lengthInBytes);
		auto db = createDeviceArray<int>(lengthInBytes);
		auto dc = createDeviceArray<int>(lengthInBytes);

		checkCudaErrors(cudaMemcpy(da, a, lengthInBytes, cudaMemcpyKind::cudaMemcpyHostToDevice));
		checkDeviceMatrix<int>(da, lengthInBytes, 1, DebugPrintArrayCount, "%d ", "da");
		checkCudaErrors(cudaMemcpy(db, b, lengthInBytes, cudaMemcpyKind::cudaMemcpyHostToDevice));
		checkDeviceMatrix<int>(db, lengthInBytes, 1, DebugPrintArrayCount, "%d ", "db");

		dim3 grid = dim3(getNumberOfParts(length, BlockSize));
		dim3 block = dim3(BlockSize);
		addClassic << <grid, block >> > (da, db, dc, length);
		checkDeviceMatrix<int>(dc, lengthInBytes, 1, DebugPrintArrayCount, "%d ", "dc");

		checkCudaErrors(cudaMemcpy(c, dc, lengthInBytes, cudaMemcpyKind::cudaMemcpyDeviceToHost));
		checkHostMatrix<int>(c, 1, 1, DebugPrintArrayCount, "%d ", "c");

		checkCudaErrors(cudaFree(da));
		checkCudaErrors(cudaFree(db));
		checkCudaErrors(cudaFree(dc));
	}

	template<typename T>
	__global__ void addWithUnroll(T *a, T *b, T *c, size_t length)
	{		
		auto offset = (blockDim.x * blockIdx.x + threadIdx.x) * ThreadsPerBlock;
		//printf("%d %d %d, offset=%d\n", blockDim.x, blockIdx.x, threadIdx.x, offset);
		//printf("offset=%d\n", offset);

#pragma unroll
		for (size_t i = 0; i < ThreadsPerBlock; i++, offset++)
		{
			if (offset >= length)
				return;
			//printf("i = %u\n", offset);
			c[offset] = a[offset] + b[offset];
		}
	}

	void runAdd2(int *a, int *b, int *c, size_t length, size_t lengthInBytes)
	{
		auto da = createDeviceArray<int>(lengthInBytes);
		auto db = createDeviceArray<int>(lengthInBytes);
		auto dc = createDeviceArray<int>(lengthInBytes);

		checkCudaErrors(cudaMemcpy(da, a, lengthInBytes, cudaMemcpyKind::cudaMemcpyHostToDevice));
		checkDeviceMatrix<int>(da, lengthInBytes, 1, DebugPrintArrayCount, "%d ", "da");
		checkCudaErrors(cudaMemcpy(db, b, lengthInBytes, cudaMemcpyKind::cudaMemcpyHostToDevice));
		checkDeviceMatrix<int>(db, lengthInBytes, 1, DebugPrintArrayCount, "%d ", "db");

		dim3 grid = dim3(getNumberOfParts(length, BlockSize * ThreadsPerBlock));
		dim3 block = dim3(BlockSize);
		addWithUnroll << <grid, block >> > (da, db, dc, length);
		checkDeviceMatrix<int>(dc, lengthInBytes, 1, DebugPrintArrayCount, "%d ", "dc");

		checkCudaErrors(cudaMemcpy(c, dc, lengthInBytes, cudaMemcpyKind::cudaMemcpyDeviceToHost));
		checkHostMatrix<int>(c, 1, 1, DebugPrintArrayCount, "%d ", "c");

		checkCudaErrors(cudaFree(da));
		checkCudaErrors(cudaFree(db));
		checkCudaErrors(cudaFree(dc));
	}
}
