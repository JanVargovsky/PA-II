#include <cudaDefs.h>
//#include "Lesson1.h"
//#include "Lesson2.h"
//#include "Lesson3.h"
//#include "Lesson4.h"
//#include "Lesson5.h"
//#include "Lesson6.h"
#include "Lesson7.h"
//#include "DifferentialEvolutionRunner.h"

cudaDeviceProp deviceProp = cudaDeviceProp();

int main(int argc, char *argv[])
{
	auto err = cudaGetLastError();
	if (err != cudaError::cudaSuccess)
	{
		printf("error: %d\n", err);
		return 1;
	}
	initializeCUDA(deviceProp);
	//lesson1::run();
	//lesson2::run();
	//lesson3::run();
	//lesson4::run();
	//lesson5::run();
	//lesson6::run(argc, argv);
	lesson7::run();

	//Project::run();
	return 0;
}
