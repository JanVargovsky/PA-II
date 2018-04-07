#pragma once
#include "DifferentialEvolutionRunner.h"

#include <functional>
#include <cudaDefs.h>
#include <curand.h>
#include <curand_kernel.h>

#pragma region Random
curandState *rs = nullptr;            //DEVICE DATA POINTER - Random number states

constexpr unsigned int TPB = 256;
constexpr unsigned int MBPTB = 4;
float *dRandomFloats = nullptr;

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

	KernelSetting ksRandom;
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
	constexpr size_t headLength = 100;
	//checkDeviceMatrix<float>(dRandomFloats, headLength * sizeof(float), 1, headLength, "%f ", "Device randomFloats (0 - 100)");
	//checkDeviceMatrix<float>(dRandomFloats, sizeInBytes, 1, length, "%f ", "Device randomFloats");
}

void randomCleanup()
{
	SAFE_DELETE_CUDA(dRandomFloats);
	SAFE_DELETE_CUDA(rs);
}

#pragma endregion

namespace Project {
	// dimension of problem (number of parameters)
	constexpr size_t D = 10; // x
	// Population size
	constexpr size_t NP = 10 * D; // y
	// differential weight, <0,2>
	constexpr float F = 0.5f;
	// crossover probability, <0,1>
	constexpr float CR = 0.5f;
	constexpr size_t Iterations = 1000;

	typedef float Type;
	template<typename T> __device__ __host__ T FitnessFunc(T *x, size_t size, size_t offset = 0);

	template<typename T>
	__global__ void KernelRandomPopulation(T *population, float *randoms, size_t offset)
	{
		int tx = blockIdx.x * blockDim.x + threadIdx.x; // D
		int ty = blockIdx.y * blockDim.y + threadIdx.y; // NP
		if (!(tx < D && ty < NP)) return;

		int index = tx * NP + ty;
		population[index] = randoms[offset + index];
	}

	template<typename T>
	__global__ void KernelNextGeneration(T* __restrict__ dInputPopulation, T* dOutputPopulation, float* dRandoms, size_t offset)
	{
		int tx = blockIdx.x * blockDim.x + threadIdx.x; // D
		int ty = blockIdx.y * blockDim.y + threadIdx.y; // NP
		if (!(tx < D && ty < NP)) return;

		int index = tx * NP + ty;

		// TODO: tx=0 calculates these values and sets them in shared memory
		// 3 random indexes - unique for each ty
		int i[3] = {
			(int)(dRandoms[offset + ty * 4 + 0] * D),
			(int)(dRandoms[offset + ty * 4 + 1] * D),
			(int)(dRandoms[offset + ty * 4 + 2] * D)
		};
		// TODO: make sure that array of 'i' is unique

		// guaranted copy parameter
		int j = (int)(dRandoms[offset + ty * 4 + 3] * D);
		// random for all
		float r = dRandoms[offset + NP * D * 4 + index];
		//printf("index=%d, x=%d, y=%d, i=[%d, %d, %d], j=%d, r=%f \n", index, tx, ty, i[0], i[1], i[2], j, r);

		if (tx == j || r < CR)
		{
			dOutputPopulation[index] = dInputPopulation[index];
		}
		else
		{
			T a = dInputPopulation[i[0]];
			T b = dInputPopulation[i[1]];
			T c = dInputPopulation[i[2]];
			dOutputPopulation[index] = c + F * (a - b);
		}

		__syncthreads();
		__shared__ T oldFitnesses[NP];
		__shared__ T newFitnesses[NP];
		if (tx == 0)
		{
			oldFitnesses[ty] = FitnessFunc(dInputPopulation, D, ty);
			newFitnesses[ty] = FitnessFunc(dOutputPopulation, D, ty);
			// printf("ty=%d, oldFitness=%f, newFitness=%f\n", ty, oldFitnesses[ty], newFitnesses[ty]);
		}
		__syncthreads();
		// if fitness is not better then keep original values
		if (newFitnesses[ty] > oldFitnesses[ty])
		{
			dOutputPopulation[index] = dInputPopulation[index];

			// DEBUG ONLY for the KernelPrintFitnesses
			if (tx == 0)
				newFitnesses[ty] = oldFitnesses[ty];
		}
	}

	template<typename T>
	__global__ void KernelPrintFitnesses(T* __restrict__ dPopulation)
	{
		int tx = blockIdx.x * blockDim.x + threadIdx.x;
		if (tx >= NP) return;
		T value = FitnessFunc(dPopulation, D, tx * D);
		printf("tx=%d, fitness=%f\n", tx, value);
	}

	template<typename T>
	__global__ void KernelParallelReduce(T* __restrict__ dPopulation, size_t size, T* __restrict__ dBest)
	{
		// TODO: parallel reduce
	}

	template<typename T>
	T* DifferentialEvolutionCalculate()
	{
		KernelSetting ksDE;
		ksDE.dimBlock = dim3(D);
		ksDE.blockSize = D;
		ksDE.dimGrid = dim3(D, NP);
		ksDE.print();

		KernelSetting ksPrintFitnesses;
		constexpr size_t printBlockSize = 256;
		ksPrintFitnesses.dimBlock = dim3(printBlockSize);
		ksPrintFitnesses.blockSize = printBlockSize;
		ksPrintFitnesses.dimGrid = dim3(getNumberOfParts(NP, printBlockSize));
		ksPrintFitnesses.print();

		size_t randomFloatsOffset = 0;

		// allocate population matrices
		T *dPopulation; // input population
		T *dPopulation2; // output population
		checkCudaErrors(cudaMalloc((void**)&dPopulation, D * NP * sizeof(T)));
		checkCudaErrors(cudaMalloc((void**)&dPopulation2, D * NP * sizeof(T)));

		// generate initial population
		KernelRandomPopulation << <ksDE.dimGrid, ksDE.dimBlock >> > (dPopulation, dRandomFloats, randomFloatsOffset);
		randomFloatsOffset += NP * D;
		cudaMemset(dPopulation2, 0, NP * D * sizeof(T));
		//checkDeviceMatrix(dPopulation, D * sizeof(T), NP, D, "%f ", "dPopulation - initial");
		//checkDeviceMatrix(dPopulation2, D * sizeof(T), NP, D, "%f ", "dPopulation2 - initial");

		printf("initial fitnesses\n");
		KernelPrintFitnesses << <ksPrintFitnesses.dimGrid, ksPrintFitnesses.dimBlock >> > (dPopulation);
		fflush(stdout);

		for (size_t i = 0; i < Iterations; i++)
		{
			printf("ITERATION = %u\n", i);

			// Generate next generation
			KernelNextGeneration << <ksDE.dimGrid, ksDE.dimBlock >> > (dPopulation, dPopulation2, dRandomFloats, randomFloatsOffset);

			//KernelPrintFitnesses << <ksPrintFitnesses.dimGrid, ksPrintFitnesses.dimBlock >> > (dPopulation);
			//checkDeviceMatrix(dPopulation, D * sizeof(T), NP, D, "%f ", "dPopulation");
			//checkDeviceMatrix(dPopulation2, D * sizeof(T), NP, D, "%f ", "dPopulation2");

			randomFloatsOffset += NP * 3; // Each candidate (NP) has 3 random indexes (indexes of candidates for mutation)
			randomFloatsOffset += NP; // One guaranteed random index for each candidate (NP)
			randomFloatsOffset += NP * D; // For all candidates and its params (CR)
			auto tmp = dPopulation;
			dPopulation = dPopulation2;
			dPopulation2 = tmp;
		}

		printf("final fitnesses\n");
		KernelPrintFitnesses << <ksPrintFitnesses.dimGrid, ksPrintFitnesses.dimBlock >> > (dPopulation);
		//checkDeviceMatrix(dPopulation, D * sizeof(T), NP, D, "%f ", "dPopulation - final");
		T *hx = new T[NP * D];
		checkCudaErrors(cudaMemcpy(hx, dPopulation, D * NP * sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		return hx;
	}

	void run()
	{
		randomInit();
		Type *result = DifferentialEvolutionCalculate<Type>();
		SAFE_DELETE_ARRAY(result);
		randomCleanup();
	}

	template<typename T>
	__device__ __host__ T SphereFunction(T *x, size_t size, size_t offset)
	{
		T result = 0;
		T *ptr = x + offset;
		for (size_t i = 0; i < size; i++, ptr++)
			result += *ptr * *ptr;
		return result;
	}

	template<typename T>
	T FitnessFunc(T *x, size_t size, size_t offset)
	{
		return SphereFunction(x, size, offset);
	}
}