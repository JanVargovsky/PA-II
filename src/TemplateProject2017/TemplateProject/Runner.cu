#include <cudaDefs.h>
//#include "Lesson1.h";
//#include "Lesson2.h";
//#include "Lesson3.h";
#include "Lesson4.h";

cudaDeviceProp deviceProp = cudaDeviceProp();

int main(int argc, char *argv[])
{
	initializeCUDA(deviceProp);
	//lesson1::run();
	//lesson2::run();
	//lesson3::run();
	lesson4::run();
	return 0;
}
