#include <cudaDefs.h>
#include <time.h>
#include <math.h>

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

const unsigned int N = 1 << 20;
const unsigned int MEMSIZE = N * sizeof(unsigned int);
const unsigned int NO_LOOPS = 100;
const unsigned int THREAD_PER_BLOCK = 256;
const unsigned int GRID_SIZE = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

void fillData(unsigned int *data, const unsigned int length)
{
	//srand(time(0));
	for (unsigned int i = 0; i < length; i++)
	{
		//data[i]= rand();
		data[i] = 1;
	}
}

void printData(const unsigned int *data, const unsigned int length)
{
	if (data == 0) return;
	for (unsigned int i = 0; i < length; i++)
	{
		printf("%u ", data[i]);
	}
	printf("\n");
}


__global__ void kernel(const unsigned int *a, const unsigned int *b, const unsigned int length, unsigned int *c)
{
	int i = blockDim.x * blockDim.x + threadIdx.x;

	if (i >= length) return;
	//TODO: Vector ADD
	c[i] = a[i] + b[i];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Tests 1. - single stream, async calling </summary>
///
/// <remarks>	16. 4. 2013. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////
void test1()
{
	unsigned int *a, *b, *c;
	unsigned int *da, *db, *dc;

	// paged-locked allocation
	cudaHostAlloc((void**)&a, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&b, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&c, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);

	fillData(a, NO_LOOPS * N);
	fillData(b, NO_LOOPS * N);

	// Data chunks on GPU
	cudaMalloc((void**)&da, MEMSIZE);
	cudaMalloc((void**)&db, MEMSIZE);
	cudaMalloc((void**)&dc, MEMSIZE);

	////TODO: create stream
	cudaStream_t stream;
	checkCudaErrors(cudaStreamCreate(&stream));

	unsigned int dataOffset = 0;
	for (int i = 0; i < NO_LOOPS; i++)
	{
		//TODO:  copy a->da, b->db
		cudaMemcpyAsync(da, &a[dataOffset], MEMSIZE, cudaMemcpyKind::cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(db, &b[dataOffset], MEMSIZE, cudaMemcpyKind::cudaMemcpyHostToDevice, stream);

		//TODO:  run the kernel in the stream
		kernel << <GRID_SIZE, THREAD_PER_BLOCK, 0, stream >> > (da, db, MEMSIZE, dc);

		//TODO:  copy dc->c
		cudaMemcpyAsync(&c[dataOffset], dc, MEMSIZE, cudaMemcpyKind::cudaMemcpyDeviceToHost, stream);

		dataOffset += N;
	}

	////TODO: Synchonize stream
	cudaStreamSynchronize(stream);
	////TODO: Destroy stream
	checkCudaErrors(cudaStreamDestroy(stream));

	printData(a, 10);
	printData(b, 10);
	printData(c, 10);

	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Tests 2. - two streams - depth first approach </summary>
///
/// <remarks>	16. 4. 2013. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////
void test2()
{
	//TODO: reuse the source code of above mentioned method test1()
	unsigned int *a, *b, *c;
	unsigned int *da, *db, *dc;

	// paged-locked allocation
	cudaHostAlloc((void**)&a, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&b, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&c, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);

	fillData(a, NO_LOOPS * N);
	fillData(b, NO_LOOPS * N);

	// Data chunks on GPU
	cudaMalloc((void**)&da, MEMSIZE);
	cudaMalloc((void**)&db, MEMSIZE);
	cudaMalloc((void**)&dc, MEMSIZE);

	////TODO: create stream
	cudaStream_t stream0;
	checkCudaErrors(cudaStreamCreate(&stream0));
	cudaStream_t stream1;
	checkCudaErrors(cudaStreamCreate(&stream1));

	unsigned int dataOffset = 0;
	for (int i = 0; i < NO_LOOPS / 2; i++)
	{
		//TODO:  copy a->da, b->db
		cudaMemcpyAsync(da, &a[dataOffset], MEMSIZE, cudaMemcpyKind::cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(db, &b[dataOffset], MEMSIZE, cudaMemcpyKind::cudaMemcpyHostToDevice, stream0);
		//TODO:  run the kernel in the stream
		kernel << <GRID_SIZE, THREAD_PER_BLOCK, 0, stream0 >> > (da, db, MEMSIZE, dc);
		//TODO:  copy dc->c
		cudaMemcpyAsync(&c[dataOffset], dc, MEMSIZE, cudaMemcpyKind::cudaMemcpyDeviceToHost, stream0);
		dataOffset += N;

		cudaMemcpyAsync(da, &a[dataOffset], MEMSIZE, cudaMemcpyKind::cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(db, &b[dataOffset], MEMSIZE, cudaMemcpyKind::cudaMemcpyHostToDevice, stream1);
		//TODO:  run the kernel in the stream
		kernel << <GRID_SIZE, THREAD_PER_BLOCK, 0, stream1 >> > (da, db, MEMSIZE, dc);
		//TODO:  copy dc->c
		cudaMemcpyAsync(&c[dataOffset], dc, MEMSIZE, cudaMemcpyKind::cudaMemcpyDeviceToHost, stream1);
		dataOffset += N;
	}

	////TODO: Synchonize stream
	cudaStreamSynchronize(stream0);
	cudaStreamSynchronize(stream1);
	////TODO: Destroy stream
	checkCudaErrors(cudaStreamDestroy(stream0));
	checkCudaErrors(cudaStreamDestroy(stream1));

	printData(a, 10);
	printData(b, 10);
	printData(c, 10);

	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Tests 3. - two streams - breadth first approach</summary>
///
/// <remarks>	Gajdi, 16. 4. 2013. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////
void test3()
{
	//TODO: reuse the source code of above mentioned method test1()
	unsigned int *a, *b, *c;
	unsigned int *da, *db, *dc;

	// paged-locked allocation
	cudaHostAlloc((void**)&a, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&b, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&c, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);

	fillData(a, NO_LOOPS * N);
	fillData(b, NO_LOOPS * N);

	// Data chunks on GPU
	cudaMalloc((void**)&da, MEMSIZE);
	cudaMalloc((void**)&db, MEMSIZE);
	cudaMalloc((void**)&dc, MEMSIZE);

	////TODO: create stream
	cudaStream_t stream0;
	checkCudaErrors(cudaStreamCreate(&stream0));
	cudaStream_t stream1;
	checkCudaErrors(cudaStreamCreate(&stream1));

	unsigned int dataOffset = 0;
	for (int i = 0; i < NO_LOOPS / 2; i++)
	{
		//TODO:  copy a->da, b->db
		cudaMemcpyAsync(da, &a[dataOffset], MEMSIZE, cudaMemcpyKind::cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(da, &a[dataOffset], MEMSIZE, cudaMemcpyKind::cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(db, &b[dataOffset], MEMSIZE, cudaMemcpyKind::cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(db, &b[dataOffset], MEMSIZE, cudaMemcpyKind::cudaMemcpyHostToDevice, stream1);

		//TODO:  run the kernel in the stream
		kernel << <GRID_SIZE, THREAD_PER_BLOCK, 0, stream0 >> > (da, db, MEMSIZE, dc);
		kernel << <GRID_SIZE, THREAD_PER_BLOCK, 0, stream1 >> > (da, db, MEMSIZE, dc);

		//TODO:  copy dc->c
		cudaMemcpyAsync(&c[dataOffset], dc, MEMSIZE, cudaMemcpyKind::cudaMemcpyDeviceToHost, stream0);
		dataOffset += N;
		cudaMemcpyAsync(&c[dataOffset], dc, MEMSIZE, cudaMemcpyKind::cudaMemcpyDeviceToHost, stream1);
		dataOffset += N;
	}

	////TODO: Synchonize stream
	cudaStreamSynchronize(stream0);
	cudaStreamSynchronize(stream1);
	////TODO: Destroy stream
	checkCudaErrors(cudaStreamDestroy(stream0));
	checkCudaErrors(cudaStreamDestroy(stream1));

	printData(a, 10);
	printData(b, 10);
	printData(c, 10);

	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);
}

int main(int argc, char *argv[])
{
	initializeCUDA(deviceProp);

	cudaEvent_t startEvent, stopEvent;
	float elapsedTime;

	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);

	cudaEventRecord(startEvent, 0);
	test1();
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	printf("Test time: %f ms\n", elapsedTime);

	cudaEventRecord(startEvent, 0);
	test2();
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	printf("Test time: %f ms\n", elapsedTime);

	cudaEventRecord(startEvent, 0);
	test3();
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	printf("Test time: %f ms\n", elapsedTime);

	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);

	return 0;
}
