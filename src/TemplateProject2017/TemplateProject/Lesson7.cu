#include "Lesson7.h"

#include <cudaDefs.h>
#include <limits>


namespace lesson7 {
	constexpr size_t BLOCK_SIZE = 8;

	template<typename T>
	void fill_data(T *array, size_t len)
	{
		for (int i = 0; i < len; i++)
			array[i] = i;

		array[len / 2] = len + 1;
	}

	template<typename T>
	__global__ void atomic_max1(T *array, size_t len, T *max)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= len) return;

		int shift = gridDim.x + blockDim.x;
		while (i < len)
		{
			if (*max < array[i])
				atomicMax(max, array[i]);
			i += shift;
		}
	}

	template<typename T>
	__global__ void atomic_max2(T *array, size_t len, T *max)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= len) return;

		int shift = gridDim.x + blockDim.x;
		T localMax = *max;
		while (i < len)
		{
			if (localMax < array[i])
				localMax = array[i];
			i += shift;
		}
		atomicMax(max, localMax);
	}

	void run()
	{
		constexpr size_t N = 500;
		int *hData = new int[N];

		cudaEvent_t start, stop;
		float time;
		createTimer(&start, &stop, &time);

		startTimer(start);
		fill_data(hData, N);
		stopTimer(start, stop, time);


		int *dData = nullptr;
		checkCudaErrors(cudaMalloc((void**)&dData, sizeof(int) * N));
		cudaMemcpy(dData, hData, sizeof(int) * N, cudaMemcpyKind::cudaMemcpyHostToDevice);

		int hMax = std::numeric_limits<int>().min();
		int *dMax = nullptr;
		checkCudaErrors(cudaMalloc((void**)&dMax, sizeof(int)));
		cudaMemcpy(dMax, &hMax, sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);

		KernelSetting ks_atomic_max;
		ks_atomic_max.dimBlock = dim3(BLOCK_SIZE);
		ks_atomic_max.blockSize = BLOCK_SIZE;
		ks_atomic_max.dimGrid = dim3(getNumberOfParts(N, ks_atomic_max.blockSize));
		startTimer(start);
		atomic_max1 << <ks_atomic_max.dimBlock, ks_atomic_max.blockSize >> > (dData, N, dMax);
		stopTimer(start, stop, time);

		startTimer(start);
		atomic_max2 << <ks_atomic_max.dimBlock, ks_atomic_max.blockSize >> > (dData, N, dMax);
		stopTimer(start, stop, time);


		delete[] hData;
	}
}