#include "DifferentialEvolutionRunner.h"
#include "DifferentialEvolution.cu"

namespace Project {

	typedef float Type;

	__device__ Type SphereFunction(Type *x, size_t size)
	{
		Type result = 0;
		Type *ptr = x;
		for (size_t i = 0; i < size; i++, ptr++)
			result += *ptr * *ptr;
		return result;
	}

	void run()
	{
		DifferentialEvolutionParameters<Type> params;
		params.D = 30;
		params.PopulationSize = 100;
		params.FitnessFunc = &SphereFunction;
		params.F = 0.5f;
		params.CR = 0.9f;

		auto de = DifferentialEvolution<Type>(params);
		auto result = de.Calculate();
	}
}