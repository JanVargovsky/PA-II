#include <cudaDefs.h>
#include "Lesson1.h";

cudaDeviceProp deviceProp = cudaDeviceProp();

int main(int argc, char *argv[])
{
	initializeCUDA(deviceProp);
	lesson1::run();
	return 0;
}
