#pragma once
#include <cudaDefs.h>
#include <functional>

namespace Project {
	template <typename T>
	struct DifferentialEvolutionParameters
	{
		// dimension of problem (number of parameters)
		size_t D;
		size_t PopulationSize;
		std::function<T(T*, size_t)> FitnessFunc;

		// differential weight, <0,2>
		float F;
		// crossover probability, <0,1>
		float CR;
	};

	template <typename T>
	class DifferentialEvolution
	{
	private:
		DifferentialEvolutionParameters<T> hParams;

	public:
		DifferentialEvolution(DifferentialEvolutionParameters<T> params) : hParams(params)
		{
		}

		T* Calculate()
		{
			return nullptr;
		}
	};
}