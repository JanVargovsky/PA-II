#pragma once
#include "DifferentialEvolutionRunner.h"

#include <cudaDefs.h>
#include <curand.h>
#include <curand_kernel.h>

#include <functional>
#include <limits>
#include <cmath>

using namespace std;

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
	//constexpr size_t headLength = 100;
	//checkDeviceMatrix<float>(dRandomFloats, headLength * sizeof(float), 1, headLength, "%f ", "Device randomFloats (0 - 100)");
	//checkDeviceMatrix<float>(dRandomFloats, sizeInBytes, 1, length, "%f ", "Device randomFloats");
}

void randomCleanup()
{
	SAFE_DELETE_CUDA(dRandomFloats);
	SAFE_DELETE_CUDA(rs);
}

#pragma endregion

constexpr size_t closest_power_of_2(const size_t x)
{
	size_t v = x;
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}

namespace Project {
	// dimension of problem (number of parameters)
	constexpr size_t D = 1000; // x
	// Population size
	constexpr size_t NP = 1000; // y
	// differential weight, <0,2>
	constexpr float F = 0.5f;
	// crossover probability, <0,1>
	constexpr float CR = 0.2f;
	constexpr size_t Iterations = 30;

	// Parallel reduce
	constexpr size_t NP2 = closest_power_of_2(NP);

	typedef float Type;
	template<typename T> __device__ __host__ T FitnessFunc(T *x, size_t size, size_t offset = 0);

	template<typename T>
	__global__ void KernelRandomPopulation(T *population, float *randoms, size_t offset)
	{
		int tx = threadIdx.x; // D
		int ty = blockIdx.x; // NP
		if (!(tx < D && ty < NP)) return;
		int index = ty * D + tx;
		//printf("tx=%d, ty=%d, index=%d\n", tx, ty, index);

		population[index] = randoms[offset + index] * 10 - 5;
		//population[index] = index;
	}

	template<typename T>
	__global__ void KernelNextGeneration(T* __restrict__ dInputPopulation, T* dOutputPopulation, float* dRandoms, size_t offset)
	{
		int tx = threadIdx.x; // D
		int ty = blockIdx.x; // NP
		if (!(tx < D && ty < NP)) return;
		int index = ty * D + tx;

		// TODO: tx=0 calculates these values and sets them in shared memory
		// 3 random indexes - unique for each ty
		int i[3] = {
			(int)(dRandoms[offset + ty * 4 + 0] * NP),
			(int)(dRandoms[offset + ty * 4 + 1] * NP),
			(int)(dRandoms[offset + ty * 4 + 2] * NP)
		};
		// TODO: make sure that array of 'i' is unique

		// guaranted copy parameter
		int j = (int)(dRandoms[offset + ty * 4 + 3] * NP);
		// random for all
		float r = dRandoms[offset + NP * D * 4 + index];
		//printf("index=%d, x=%d, y=%d, i=[%d, %d, %d], j=%d, r=%f \n", index, tx, ty, i[0], i[1], i[2], j, r);

		if (tx == j || r < CR)
		{
			dOutputPopulation[index] = dInputPopulation[index];
		}
		else
		{
			T a = dInputPopulation[ty * D + i[0]];
			T b = dInputPopulation[ty * D + i[1]];
			T c = dInputPopulation[ty * D + i[2]];
			dOutputPopulation[index] = c + F * (a - b);
		}

		__shared__ T sOldFitnesses[NP];
		__shared__ T sNewFitnesses[NP];
		__syncthreads();
		if (tx == 0)
		{
			sOldFitnesses[ty] = FitnessFunc(dInputPopulation, D, ty * D);
			sNewFitnesses[ty] = FitnessFunc(dOutputPopulation, D, ty * D);
			//printf("ty=%d, oldFitness=%f, newFitness=%f\n", ty, sOldFitnesses[ty], sNewFitnesses[ty]);
		}
		__syncthreads();
		// if fitness is not better then keep original values
		if (sNewFitnesses[ty] > sOldFitnesses[ty])
		{
			//if (tx == 0) printf("no better value at ty=%d, oldFitness=%f, newFitness=%f\n", ty, sOldFitnesses[ty], sNewFitnesses[ty]);
			//printf("rollback value at %d, index=%d, from=%f, to=%f\n", ty, index, dOutputPopulation[index], dInputPopulation[index]);
			dOutputPopulation[index] = dInputPopulation[index];
		}
		else
		{
			//if (tx == 0 && sNewFitnesses[ty] < sOldFitnesses[ty])
			//	printf("better fitness found at ty=%d, oldFitness=%f, newFitness=%f\n", ty, sOldFitnesses[ty], sNewFitnesses[ty]);
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

	constexpr int int_max = numeric_limits<int>().max();
	constexpr Type type_max = numeric_limits<Type>().max();

	template<typename T>
	__global__ void KernelParallelReduce(T* __restrict__ dPopulation, size_t* __restrict__ dIndexOfBest)
	{
		__shared__ T sFitnesses[NP2];
		__shared__ int sIndexes[NP2];

		int tid = threadIdx.x;

		if (tid >= NP2)
			return;

		// global memory -> shared memory
		if (tid >= NP)
		{
			sIndexes[tid] = int_max;
			sFitnesses[tid] = type_max;
		}
		else
		{
			sIndexes[tid] = tid;
			sFitnesses[tid] = FitnessFunc(dPopulation, D, tid * D);
		}
		__syncthreads();

		//printf("tid=%d, index=%d, fitness=%f\n", tid, sIndexes[tid], sFitnesses[tid]);

		if (NP2 >= 2048) // compile time
		{
			if (tid >= 1024) return;
			if (sFitnesses[tid] > sFitnesses[tid + 1024])
			{
				sFitnesses[tid] = sFitnesses[tid + 1024];
				sIndexes[tid] = sIndexes[tid + 1024];
			}
			__syncthreads();
		}

		if (NP2 >= 1024) // compile time
		{
			if (tid >= 512) return;
			if (sFitnesses[tid] > sFitnesses[tid + 512])
			{
				sFitnesses[tid] = sFitnesses[tid + 512];
				sIndexes[tid] = sIndexes[tid + 512];
			}
			__syncthreads();
		}

		if (NP2 >= 512) // compile time
		{
			if (tid >= 256) return;
			if (sFitnesses[tid] > sFitnesses[tid + 256])
			{
				sFitnesses[tid] = sFitnesses[tid + 256];
				sIndexes[tid] = sIndexes[tid + 256];
			}
			__syncthreads();
		}

		if (NP2 >= 256) // compile time
		{
			if (tid >= 128) return;
			if (sFitnesses[tid] > sFitnesses[tid + 128])
			{
				sFitnesses[tid] = sFitnesses[tid + 128];
				sIndexes[tid] = sIndexes[tid + 128];
			}
			__syncthreads();
		}

		if (NP2 >= 128) // compile time
		{
			if (tid >= 64) return;
			if (sFitnesses[tid] > sFitnesses[tid + 64])
			{
				sFitnesses[tid] = sFitnesses[tid + 64];
				sIndexes[tid] = sIndexes[tid + 64];
			}
			__syncthreads();
		}

		// tid < 32
		if (NP2 >= 64) // compile time
		{
			if (tid >= 32) return;
			if (sFitnesses[tid] > sFitnesses[tid + 32]) {
				sFitnesses[tid] = sFitnesses[tid + 32];
				sIndexes[tid] = sIndexes[tid + 32];
			}
		}
		if (NP2 >= 32) // compile time
		{
			if (tid >= 16) return;
			if (sFitnesses[tid] > sFitnesses[tid + 16]) {
				sFitnesses[tid] = sFitnesses[tid + 16];
				sIndexes[tid] = sIndexes[tid + 16];
			}
		}
		if (NP2 >= 16) // compile time
		{
			if (tid >= 8) return;
			if (sFitnesses[tid] > sFitnesses[tid + 8]) {
				sFitnesses[tid] = sFitnesses[tid + 8];
				sIndexes[tid] = sIndexes[tid + 8];
			}
		}
		if (NP2 >= 8) // compile time
		{
			if (tid >= 4) return;
			if (sFitnesses[tid] > sFitnesses[tid + 4]) {
				sFitnesses[tid] = sFitnesses[tid + 4];
				sIndexes[tid] = sIndexes[tid + 4];
			}
		}
		if (NP2 >= 4) // compile time
		{
			if (tid >= 2) return;
			if (sFitnesses[tid] > sFitnesses[tid + 2]) {
				sFitnesses[tid] = sFitnesses[tid + 2];
				sIndexes[tid] = sIndexes[tid + 2];
			}
		}
		if (NP2 >= 2) // compile time
		{
			if (tid >= 1) return;
			if (sFitnesses[tid] > sFitnesses[tid + 1]) {
				sFitnesses[tid] = sFitnesses[tid + 1];
				sIndexes[tid] = sIndexes[tid + 1];
			}
		}
		//if (tid == 0)
		*dIndexOfBest = sIndexes[0];
		printf("Best fitness index=%d, fitness=%f\n", sIndexes[0], sFitnesses[0]);
		//for (int i = 0; i < NP; i++)
			//printf("i=%d, index=%d, fitness=%f\n", i, sIndexes[i], sFitnesses[i]);
	}

	template<typename T>
	void PrintPopulationWithFitnesses(T* dPopulation)
	{
		T *population = nullptr;
		checkCudaErrors(cudaHostAlloc((void**)&population, NP * D * sizeof(T), cudaHostAllocWriteCombined));
		checkCudaErrors(cudaMemcpy(population, dPopulation, NP * D * sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		for (size_t i = 0; i < NP; i++)
		{
			printf("i=%u, params=[", i);
			for (size_t d = 0; d < D; d++)
			{
				if (d != 0) printf(", ");
				printf("%f", population[i * D + d]);
			}
			T fitness = FitnessFunc(population, D, i * D);
			printf("] fitness=%f\n", fitness);
		}

		cudaFreeHost(population);
	}

	template<typename T>
	void PrintPopulationWithFitnesses(T* dNewPopulation, T* dOldPopulation)
	{
		T *newPopulation = nullptr;
		checkCudaErrors(cudaHostAlloc((void**)&newPopulation, NP * D * sizeof(T), cudaHostAllocWriteCombined));
		checkCudaErrors(cudaMemcpy(newPopulation, dNewPopulation, NP * D * sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		T *oldPopulation = nullptr;
		checkCudaErrors(cudaHostAlloc((void**)&oldPopulation, NP * D * sizeof(T), cudaHostAllocWriteCombined));
		checkCudaErrors(cudaMemcpy(oldPopulation, dOldPopulation, NP * D * sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		for (size_t i = 0; i < NP; i++)
		{
			printf("i=%u, params=[", i);
			for (size_t d = 0; d < D; d++)
			{
				if (d != 0) printf(", ");
				printf("%f", newPopulation[i * D + d]);
			}
			T newFitness = FitnessFunc(newPopulation, D, i * D);
			T oldFitness = FitnessFunc(oldPopulation, D, i * D);
			printf("] newFitness=%f, oldFitness=%f\n", newFitness, oldFitness);
		}

		cudaFreeHost(newPopulation);
		cudaFreeHost(oldPopulation);
	}

	template<typename T>
	void PrintPopulationsWithFitnesses(T* dOldPopulation, T* dNewPopulation)
	{
		T *oldPopulation = nullptr;
		checkCudaErrors(cudaHostAlloc((void**)&oldPopulation, NP * D * sizeof(T), cudaHostAllocWriteCombined));
		checkCudaErrors(cudaMemcpy(oldPopulation, dOldPopulation, NP * D * sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		T *newPopulation = nullptr;
		checkCudaErrors(cudaHostAlloc((void**)&newPopulation, NP * D * sizeof(T), cudaHostAllocWriteCombined));
		checkCudaErrors(cudaMemcpy(newPopulation, dNewPopulation, NP * D * sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		for (size_t i = 0; i < NP; i++)
		{
			printf("i=%u\n", i);

			printf("old = [");
			for (size_t d = 0; d < D; d++)
			{
				if (d != 0) printf(", ");
				printf("%f", oldPopulation[i * D + d]);
			}
			T oldFitness = FitnessFunc(oldPopulation, D, i * D);
			printf("] oldFitness=%f\n", oldFitness);

			printf("new = [");
			for (size_t d = 0; d < D; d++)
			{
				if (d != 0) printf(", ");
				printf("%f", newPopulation[i * D + d]);
			}
			T newFitness = FitnessFunc(newPopulation, D, i * D);
			printf("] newFitness=%f\n", newFitness);
		}

		cudaFreeHost(newPopulation);
		cudaFreeHost(oldPopulation);
	}

	template<typename T>
	T* DifferentialEvolutionCalculate()
	{
		KernelSetting ksDE;
		ksDE.dimBlock = dim3(D);
		ksDE.blockSize = D;
		ksDE.dimGrid = dim3(NP);
		ksDE.print();

		KernelSetting ksPrintFitnesses;
		constexpr size_t printBlockSize = 256;
		ksPrintFitnesses.dimBlock = dim3(printBlockSize);
		ksPrintFitnesses.blockSize = printBlockSize;
		ksPrintFitnesses.dimGrid = dim3(getNumberOfParts(NP, printBlockSize));
		ksPrintFitnesses.print();

		KernelSetting ksParallelReduce;
		ksParallelReduce.dimBlock = dim3(NP2);
		ksParallelReduce.blockSize = NP2;
		ksParallelReduce.dimGrid = dim3();
		ksParallelReduce.print();

		size_t randomFloatsOffset = 0;

		// allocate population matrices
		T *dPopulation = nullptr; // input population
		T *dPopulation2 = nullptr; // output population
		checkCudaErrors(cudaMalloc((void**)&dPopulation, D * NP * sizeof(T)));
		checkCudaErrors(cudaMalloc((void**)&dPopulation2, D * NP * sizeof(T)));

		// generate initial population
		checkDeviceMatrix(dPopulation, D * sizeof(T), 2, D, "%f ", "dPopulation - initial");
		KernelRandomPopulation << <ksDE.dimGrid, ksDE.dimBlock >> > (dPopulation, dRandomFloats, randomFloatsOffset);
		checkError();
		//checkDeviceMatrix(dPopulation, D * sizeof(T), NP, D, "%f ", "dPopulation - initial");
		checkDeviceMatrix(dPopulation, D * sizeof(T), 2, D, "%f ", "dPopulation - initial");

		//randomFloatsOffset += NP * D;
		randomFloatsOffset++;
		cudaMemset(dPopulation2, 0, NP * D * sizeof(T));
		//checkDeviceMatrix(dPopulation2, D * sizeof(T), NP, D, "%f ", "dPopulation2 - initial");
		size_t *dIndexOfBest = nullptr;
		checkCudaErrors(cudaMalloc((void**)&dIndexOfBest, sizeof(size_t)));
		size_t *hIndexOfBest = new size_t;

		//printf("initial fitnesses\n");
		//KernelPrintFitnesses << <ksParallelReduce.dimGrid, ksParallelReduce.dimBlock >> > (dPopulation);
		KernelParallelReduce << <ksParallelReduce.dimGrid, ksParallelReduce.dimBlock >> > (dPopulation, dIndexOfBest);
		checkError();
		checkCudaErrors(cudaMemcpy(hIndexOfBest, dIndexOfBest, sizeof(size_t), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		//PrintPopulationWithFitnesses(dPopulation);

		for (size_t i = 0; i < Iterations; i++)
		{
			printf("ITERATION = %u\n", i);

			// Generate next generation
			KernelNextGeneration << <ksDE.dimGrid, ksDE.dimBlock >> > (dPopulation, dPopulation2, dRandomFloats, randomFloatsOffset);
			checkError();
			KernelParallelReduce << <ksParallelReduce.dimGrid, ksParallelReduce.dimBlock >> > (dPopulation2, dIndexOfBest);
			checkError();
			checkCudaErrors(cudaMemcpy(hIndexOfBest, dIndexOfBest, sizeof(size_t), cudaMemcpyKind::cudaMemcpyDeviceToHost));
			printf("current best fitness is at %u\n", *hIndexOfBest);

			//PrintPopulationsWithFitnesses(dPopulation, dPopulation2);
			//KernelPrintFitnesses << <ksPrintFitnesses.dimGrid, ksPrintFitnesses.dimBlock >> > (dPopulation2);
			//checkDeviceMatrix(dPopulation, D * sizeof(T), NP, D, "%f ", "dPopulation");
			//checkDeviceMatrix(dPopulation2, D * sizeof(T), NP, D, "%f ", "dPopulation2");


			//randomFloatsOffset += NP * 3; // Each candidate (NP) has 3 random indexes (indexes of candidates for mutation)
			//randomFloatsOffset += NP; // One guaranteed random index for each candidate (NP)
			//randomFloatsOffset += NP * D; // For all candidates and its params (CR)
			randomFloatsOffset++;
			auto tmp = dPopulation;
			dPopulation = dPopulation2;
			dPopulation2 = tmp;
		}

		printf("final fitnesses\n");
		//KernelPrintFitnesses << <ksParallelReduce.dimGrid, ksParallelReduce.dimBlock >> > (dPopulation);
		//checkDeviceMatrix(dPopulation, D * sizeof(T), NP, D, "%f ", "dPopulation - final");
		KernelParallelReduce << <ksParallelReduce.dimGrid, ksParallelReduce.dimBlock >> > (dPopulation, dIndexOfBest);
		checkCudaErrors(cudaMemcpy(hIndexOfBest, dIndexOfBest, sizeof(size_t), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		T *hx = new T[D];
		T *dPopulationPtr = dPopulation + (*hIndexOfBest * D);
		checkCudaErrors(cudaMemcpy(hx, dPopulationPtr, D * sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		SAFE_DELETE_CUDA(dPopulation);
		SAFE_DELETE_CUDA(dPopulation2);
		SAFE_DELETE_CUDA(dIndexOfBest);
		SAFE_DELETE(hIndexOfBest);
		return hx;
	}

	void run()
	{
		randomInit();
		Type *result = DifferentialEvolutionCalculate<Type>();
		printf("[");
		for (size_t i = 0; i < D; i++)
		{
			if (i != 0)
				printf(", ");
			printf("%f", result[i]);
		}
		printf("] = %f", FitnessFunc(result, D));
		SAFE_DELETE_ARRAY(result);
		randomCleanup();
	}

	template<typename T>
	inline __device__ __host__ T SphereFunction(T *x, size_t size, size_t offset)
	{
		T result = 0;
		T *ptr = x + offset;
		for (size_t i = 0; i < size; i++, ptr++)
			result += *ptr * *ptr;
		return result;
	}

	template<typename T>
	inline __device__ __host__ T RastriginFunction(T *x, size_t size, size_t offset)
	{
		constexpr T A = 10;
		constexpr double PI = 3.141592653589793238463;

		T result = A * size;
		T *ptr = x + offset;
		for (size_t i = 0; i < size; i++, ptr++)
			result += (*ptr * *ptr - A * cos(2 * PI * *ptr));
		return result;
	}

	template<typename T>
	T FitnessFunc(T *x, size_t size, size_t offset)
	{
		//return SphereFunction(x, size, offset);
		return RastriginFunction(x, size, offset);
	}
}