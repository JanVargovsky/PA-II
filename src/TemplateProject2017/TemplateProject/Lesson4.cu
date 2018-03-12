#include <cudaDefs.h>
#include <time.h>
#include <math.h>
#include <random>

namespace lesson4 {
	using namespace std;

	cudaError_t error = cudaSuccess;
	cudaDeviceProp deviceProp = cudaDeviceProp();

	struct FooBar {
		int foo;
		float bar;
	};

	__constant__ __device__ int dScalar;
	__constant__ __device__ FooBar dFooBar;
	const int FooBarsCount = 5;
	__constant__ __device__ FooBar dFooBars[FooBarsCount];

	__global__ void kernelScalar()
	{
		printf("Device scalar: %d\n", dScalar);
	}

	void runScalar()
	{
		int hScalarValue = 5;
		checkCudaErrors(cudaMemcpyToSymbol(static_cast<const void*>(&dScalar), static_cast<const void*>(&hScalarValue), 1 * sizeof(int)));

		kernelScalar << <1, 1 >> > ();

		int hScalarValue2 = -1;
		checkCudaErrors(cudaMemcpyFromSymbol(static_cast<void*>(&hScalarValue2), static_cast<const void*>(&dScalar), 1 * sizeof(int)));

		printf("Scalar: %d -> %d\n", hScalarValue, hScalarValue2);
	}

	__global__ void kernelStruct()
	{
		printf("Device struct: (%d, %f)\n", dFooBar.foo, dFooBar.bar);

	}

	void runStruct()
	{
		FooBar hFooBarValue{ 42, 0.42f };
		checkCudaErrors(cudaMemcpyToSymbol(static_cast<const void*>(&dFooBar), static_cast<const void*>(&hFooBarValue), 1 * sizeof(FooBar)));

		kernelStruct << <1, 1 >> > ();

		FooBar hFooBarValue2;
		checkCudaErrors(cudaMemcpyFromSymbol(static_cast<void*>(&hFooBarValue2), static_cast<const void*>(&dFooBar), 1 * sizeof(FooBar)));

		printf("Struct (%d, %f) -> (%d, %f)\n", hFooBarValue.foo, hFooBarValue.bar, hFooBarValue2.foo, hFooBarValue2.bar);
	}

	__global__ void kernelArrayOfStructs()
	{
		int i = threadIdx.x;
		printf("Device struct[%d]: (%d, %f)\n", i, dFooBars[i].foo, dFooBars[i].bar);
	}

	void runArrayOfStructs()
	{
		FooBar *hFooBarValues = new FooBar[FooBarsCount];
		for (int i = 0; i < FooBarsCount; i++)
		{
			hFooBarValues[i].foo = 42 + i;
			hFooBarValues[i].bar = 0.42f + i;
		}
		checkCudaErrors(cudaMemcpyToSymbol(static_cast<const void*>(dFooBars), static_cast<const void*>(hFooBarValues), FooBarsCount * sizeof(FooBar)));

		kernelArrayOfStructs << <1, FooBarsCount >> > ();

		FooBar *hFooBarValues2 = new FooBar[FooBarsCount];
		checkCudaErrors(cudaMemcpyFromSymbol(static_cast<void*>(hFooBarValues2), static_cast<const void*>(dFooBars), FooBarsCount * sizeof(FooBar)));
	}

	void run()
	{
		runScalar();
		runStruct();
		runArrayOfStructs();
	}
}