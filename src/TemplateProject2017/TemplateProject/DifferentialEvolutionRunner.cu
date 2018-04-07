#pragma once
#include "DifferentialEvolutionRunner.h"

#include <functional>
#include <cudaDefs.h>
#include <curand.h>
#include <curand_kernel.h>

#pragma region Random
curandState *rs = nullptr;            //DEVICE DATA POINTER - Random number states
KernelSetting ksRandom;

constexpr unsigned int TPB = 256;
constexpr unsigned int MBPTB = 4;
float *dRandomFloats = nullptr;
//float *dRandomDimensions = nullptr;

__global__ void initRandomStates(curandState *rs, const unsigned long seed)
{
	unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
	curandState *r = rs + offset;            //Thread data pointer offset
	curand_init(seed, offset, 0, r);
}

__global__ void initRandomFloats(curandState* __restrict__ rs, const unsigned int length, float* data)
{
	unsigned int offset = threadIdx.x + blockIdx.x * TPB * MBPTB;
	unsigned int rsOffset = threadIdx.x + blockIdx.x * TPB;

	curandState *trs = &rs[rsOffset];

#pragma unroll MBPTB
	for (unsigned int i = 0; i < MBPTB; i++)
	{
		if (offset >= length) return;
		data[offset] = curand_uniform(trs);
		offset += TPB;
	}
}

void randomInit()
{
	constexpr unsigned int length = 1 << 20;
	constexpr unsigned int sizeInBytes = length * sizeof(float);
	constexpr unsigned long seed = 42;

	ksRandom.blockSize = TPB;
	ksRandom.noChunks = MBPTB;
	ksRandom.dimBlock = dim3(TPB, 1, 1);
	ksRandom.dimGrid = getNumberOfParts(length, TPB * MBPTB);
	ksRandom.print();

	//Random Sates
	checkCudaErrors(cudaMalloc((void**)&rs, ksRandom.dimGrid.x * ksRandom.dimBlock.x * sizeof(curandState)));
	initRandomStates << <ksRandom.dimGrid, ksRandom.dimBlock >> > (rs, seed);

	//Init random numbers
	checkCudaErrors(cudaMalloc((void**)&dRandomFloats, sizeInBytes));

	initRandomFloats << <ksRandom.dimGrid, ksRandom.dimBlock >> > (rs, length, dRandomFloats);
	checkDeviceMatrix<float>(dRandomFloats, sizeInBytes, 1, length, "%f ", "Device randomFloats");
}

void randomCleanup()
{
	SAFE_DELETE_CUDA(dRandomFloats);
	SAFE_DELETE_CUDA(rs);
}

#pragma endregion

namespace Project {
	template <typename T>
	struct DifferentialEvolutionParameters
	{
		// dimension of problem (number of parameters)
		size_t D;
		// Population size
		size_t N;

		std::function<T(T*, size_t)> FitnessFunc;

		// differential weight, <0,2>
		float F;
		// crossover probability, <0,1>
		float CR;
	};

	typedef float Type;

	template<typename T>
	__device__ T SphereFunction(T *x, size_t size)
	{
		T result = 0;
		T *ptr = x;
		for (size_t i = 0; i < size; i++, ptr++)
			result += *ptr * *ptr;
		return result;
	}

	template<typename T>
	T* DifferentialEvolutionCalculate(DifferentialEvolutionParameters<T> &params)
	{
		return nullptr;
	}

	void run()
	{
		DifferentialEvolutionParameters<Type> params;
		params.D = 30;
		params.N = 100;
		params.FitnessFunc = &SphereFunction<Type>;
		params.F = 0.5f;
		params.CR = 0.9f;

		randomInit();
		Type *result = DifferentialEvolutionCalculate(params);
		randomCleanup();
	}
}