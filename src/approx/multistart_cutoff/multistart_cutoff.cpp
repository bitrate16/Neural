#include <iostream>
#include <string>
#include <limits>
#include <cstring>
#include <cmath>
#include <chrono>

#include "NetworkTestingCommon.h"

#ifdef ENABLE_PRINT
	#define PRINT_BOOL 1
#else
	#define PRINT_BOOL 0
#endif

// #define DEBUG_CUT_SET

// g++ -O3 src/approx/multistart_cutoff/multistart_cutoff.cpp -o bin/multistart_cutoff -Iinclude -lstdc++fs && ./bin/multistart_cutoff

// ./bin/generate_set 0.0 1.0 100000 "sin(t * 2.0 * 3.14) * 0.5 + 0.5" input/train.nse
// ./bin/generate_set 0.0 1.0 100 "sin(t * 2.0 * 3.14) * 0.5 + 0.5" input/test.nse
// ./bin/multistart_cutoff 4 1 3 3 1 10.0 TanH input/teach.nse input/test.nse output/mc_network.neetwook

// g++ -O3 src/approx/multistart_cutoff/multistart_cutoff.cpp -o bin/multistart_cutoff -Iinclude -lstdc++fs && ./bin/multistart_cutoff 4 1 3 3 1 10.0 TanH input/train.nse input/test.nse output/mc_network.neetwook

int main(int argc, char** argv) {
	
	// The source for this algorithm is based on < - - insert diploma here - - > [1]
	
	// This example does multistart cutoff algorighm.
	// Example of teaching neural network to approximate function Y = F(X)
	//  Where X, Y are scalar.
	// 1. Generating set of networks (generate_random_weight_networks).
	// > Refering to [1], total amount of networks in the networks set is A = W * n,
	// >  where W - total amount of weights, n - total amount of input neuroms.
	// > Proctically, than more neurons network has, than more variations of weights may 
	// >  occur, and than more local minimas must be excluded during the training.
	// > In this case the primitive algorighm is used:
	// >  Each network in set is being initialized with random set of weights distributed on [0.5, 0.5].
	// >  Next update step is to implement k-means method to split weights into k 
	// >   disjoint point classes and generate weights for each class.
	// 
	// 2. Read train set & split into set of subsets (split_linear_set) with sizes of Ci.
	// > Primitive set used in comparison between standart backpropagation and M-C.
	// >  The task of the network is to approximate the given function and 
	// >   minimize the error value for given topology.
	// >  Basing on current task, the methosics of set generation is splitting 
	// >   function describe area on line into R sectors, combine (X, Y) pairs 
	// >   and randomly shuffle. Such method is used to generate common 
	// >   test / teach set for both teach algorithms.
	// 
	// 3. For all left sets on step i perform Ci steps of learning 
	//     (learn network on Ci examples) using local-optimal algorithm.
	//     Current implementation is backpropagation algorithm.
	// > Referring to [1], total amount of steps in the algorithm is N = [log2(A)].
	// > Each step network is being reduced by Ai / 2, where Ai - amount of starts, 
	// >  left from the previous step.
	// > Each step amount of learn iterations is inversely to the amount of the left networks.
	// > Ai = [A / 2^i] is the number of networks left on i step.
	// > C1 = C / (2 ^ [log2(A)] - 1) is the size of learning set on 1st step.
	// > Ci = C1 * 2^(i-1) is the size of learning set on i step. Ci must grow on i growing.
	// > Referring to [1], to reduce amount of overlearning the network, Ci -= Ci / [log2(A)].
	// > Referring to [1], next update is to implement network set grouping 
	// >  and levenberg-marquardt learning algorithm.
	// 
	// 4. For all networks calculate gradient of error value V, average error value on set E.
	// > V is being calculated as average square error value difference of 
	// >  error values before teach and after teach.
	// >  V<b>i = SUM [ output<b> - test ]i ^ 2 / OUT_SIZE
	// >  V<a>i = SUM [ output<a> - test ]i ^ 2 / OUT_SIZE
	// >  V = (V<b> - V<a>) / SET_SIZE
	// > E is calculated as average linear error value during test
	// 
	// 5. Sort all networks ascending by V.
	//    Sort all networks descending by E.
	// > As the result we get a set of networks with Maximal E and minimal V.
	// 
	// 6. Select Ai networks and remove them from set.
	// 
	// 7. Repeat until N.
	//
	// 8. Left network is the resulting network that has mininal 
	//     error value for current topology.
	
	// > generate_set.cpp      - generates distributed over line and randomly shuffle
	// > generate_networks.cpp - random networks generator
	// > train_network.cpp     - trains selected network
	// > split_set.cpp         - splits given set into several subsets
	// > calculate_mean.cpp    - calculate error value on passed network 
	
	// Input:
	// 1. layers count [L].
	// 2+i. layer i size.
	// 2+L. weights dispersion.
	// 3+L. activator function (TanH, Sigmoid, Linear).
	// 4+L. input train set.
	// 5+L. input test set.
	// 6+L. output file for network
	
	// Output: error value for test set on produced network.
	
	
	// 0. Parse input
	if (argc < 2) {
		std::cout << "Not enough arguments" << std::endl;
		return 0;
	}
	
	int L = std::stoi(argv[1]);
	
	if (argc < L + 6)  {
		std::cout << "Not enough arguments" << std::endl;
		return 0;
	}
	
	std::vector<int> dimensions(L);
	
	for (int i = 0; i < dimensions.size(); ++i)
		dimensions[i] = std::stoi(argv[i + 2]);
	
	double Wd = std::stod(argv[L + 2]);
	
	NNSpace::ActivatorType activator;
	if (strcmp("TanH", argv[L + 3]) == 0)
		activator = NNSpace::ActivatorType::TANH;
	else if (strcmp("Sigmoid", argv[L + 3]) == 0)
		activator = NNSpace::ActivatorType::SIGMOID;
	else if (strcmp("Linear", argv[L + 3]) == 0)
		activator = NNSpace::ActivatorType::LINEAR;
	else {
		std::cout << "Invalid activator type" << std::endl;
		return 0;
	}
	
	std::string train_set_file = argv[L + 4];
	std::string test_set_file = argv[L + 5];
	std::string output_filename = argv[L + 6];
	
	
	// 1. Read train & test set
	std::vector<NNSpace::linear_set_point> train_set;
	if (!NNSpace::read_linear_set(train_set, train_set_file, 0, PRINT_BOOL)) {
		std::cout << "Failed reading train set" << std::endl;
		return 0;
	}
	
	std::vector<NNSpace::linear_set_point> test_set;
	if (!NNSpace::read_linear_set(test_set, test_set_file, 0, PRINT_BOOL)) {
		std::cout << "Failed reading test set" << std::endl;
		return 0;
	}
	
	/*
	// Calculate network weight dispersion as (max_test_y - min_test_y) * (max_test_x - min_test_x) * 4
	long double Wd = 0.0;
	
	{
		long double minx = std::numeric_limits<long double>::max();
		long double maxx = 0.0;
		long double miny = std::numeric_limits<long double>::max();
		long double maxy = 0.0;
		
		for (int i = 0; i < train_set.size(); ++i) {
			if (train_set[i].x > maxx)
				maxx = train_set[i].x;
			if (train_set[i].x < minx)
				minx = train_set[i].x;
			if (train_set[i].y > maxy)
				maxy = train_set[i].y;
			if (train_set[i].y < miny)
				miny = train_set[i].y;
		}
		
		Wd = (maxy - miny + maxx - minx) * 10.0;
	}
	*/
	
	// 2. Generate train subsets
	int A = 0;
	for (int i = 0; i < L - 1; ++i) 
		A += dimensions[i] * dimensions[i + 1];
	
	std::vector<std::vector<NNSpace::linear_set_point>> train_set_set;
	NNSpace::split_linear_set_base_2(train_set_set, train_set, A, 0, PRINT_BOOL);
	
	// -- Time record start here --
	auto timestamp_1 = std::chrono::high_resolution_clock::now();
	unsigned long learn_iterations_count = 0;
	
	// 3. Generate Networks
	std::vector<NNSpace::MLNetwork> networks;
	generate_random_weight_networks(networks, dimensions, activator, Wd, 0, PRINT_BOOL);
	
	std::vector<int> index_array;
	std::vector<int> half_index_array;
	
	std::vector<long double> V;
	std::vector<long double> E;
	
	long double E_min, V_max;
		
	// 4. Loop
	int step = 0;
	int N = NNSpace::log2(A);
	
	while (step < N) {
		
		index_array.resize(networks.size());
		half_index_array.resize(networks.size() / 2);
		for (int i = 0; i < index_array.size(); ++i)
			index_array[i] = i;
		
		V.resize(networks.size());
		E.resize(networks.size());
		
		E_min = std::numeric_limits<long double>::max();
		V_max = 0.0;
			
		// Collect statistics
		learn_iterations_count += networks.size() * train_set_set[step].size();
		
		for (int i = 0; i < networks.size(); ++i) {
			// 5. Calculate error value for each network now (V<b>)
			
			V[i] = NNSpace::calculate_linear_error(networks[i], test_set, PRINT_BOOL);
			
			// 6. Perform teaching of all networks
			NNSpace::train_network_backpropagation(networks[i], train_set_set[step], 1, PRINT_BOOL);
			
			// 8. Calculate Average error value now (E)
			E[i] = NNSpace::calculate_linear_error(networks[i], test_set, PRINT_BOOL);
			if (E[i] < E_min)
				E_min = E[i];
			
			// 7. Calculate error value for each network now (V<a>)
			V[i] = E[i] - V[i];
			
			// Calculate V
			V[i] /= (long double) test_set.size();
			if (V[i] > V_max)
				V_max = V[i];
		}
		
#ifdef DEBUG_CUT_SET
		std::cout << std::endl;
		std::cout << "Indexes ordered by distance before sort (id, distance): " << std::endl;
		for (int i = 0; i < index_array.size(); ++i)
			std::cout << '(' << index_array[i] << ", " << (V_max - V[index_array[i]] + E[index_array[i]] - E_min) << ") " << std::endl;
		std::cout << std::endl;
#endif
		
		// 8. Order network indexes
		std::sort(index_array.begin(), index_array.end(), [&V, &E, &V_max, &E_min](const int& a, const int& b) {
			return 	(V_max - V[a] + E[a] - E_min)  // Distance from A to error values
					>
					(V_max - V[b] + E[b] - E_min); // Distance from B to error values
		});
		
#ifdef DEBUG_CUT_SET
		std::cout << "Indexes ordered by distance after sort (id, distance): " << std::endl;
		for (int i = 0; i < index_array.size(); ++i)
			std::cout << '(' << index_array[i] << ", " << (V_max - V[index_array[i]] + E[index_array[i]] - E_min) << ") " << std::endl;
		std::cout << std::endl;
#endif
		
		// 9. Select Ai networks and remove
		half_index_array.assign(index_array.begin(), index_array.begin() + index_array.size() / 2 + index_array.size() % 2);
		
#ifdef DEBUG_CUT_SET
		std::cout << "Indexes ordered by distance half before sort (id, distance): " << std::endl;
		for (int i = 0; i < half_index_array.size(); ++i)
			std::cout << '(' << half_index_array[i] << ", " << (V_max - V[half_index_array[i]] + E[half_index_array[i]] - E_min) << ") " << std::endl;
		std::cout << std::endl;
#endif
		
		std::sort(half_index_array.begin(), half_index_array.end(), [](const int& a, const int& b) { return a > b; });
		
#ifdef DEBUG_CUT_SET
		std::cout << "Indexes ordered by distance half (id, distance): " << std::endl;
		for (int i = 0; i < half_index_array.size(); ++i)
			std::cout << '(' << half_index_array[i] << ", " << (V_max - V[half_index_array[i]] + E[half_index_array[i]] - E_min) << ") " << std::endl;
		std::cout << std::endl;
#endif
		
		for (int i = 0; i < half_index_array.size(); ++i)
			networks.erase(networks.begin() + half_index_array[i]);
		
		++step;
	}
	
	// -- Time record stop here --
	auto timestamp_2 = std::chrono::high_resolution_clock::now();

	// 10. Calculate resulting network error value & print
	long double error_value_result = NNSpace::calculate_linear_error(networks[0], test_set, PRINT_BOOL);
	
	std::cout << "Result error value: " << error_value_result << std::endl;
	std::cout << "Result learning time: " << std::chrono::duration_cast<std::chrono::milliseconds>(timestamp_2 - timestamp_1).count() << "ms" << std::endl;
	std::cout << "Result train iterations: " << learn_iterations_count << std::endl;
	std::cout << "Serializing into: " << output_filename << std::endl;
	NNSpace::store_network(networks[0], output_filename, PRINT_BOOL);
	
	return 0;
};