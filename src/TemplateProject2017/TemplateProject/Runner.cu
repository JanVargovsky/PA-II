#include <cudaDefs.h>
//#include "Lesson1.h";
#include "Lesson2.h";

cudaDeviceProp deviceProp = cudaDeviceProp();

int main(int argc, char *argv[])
{
	initializeCUDA(deviceProp);
	//lesson1::run();
	lesson2::run();
	return 0;
}
