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

// g++ -O3 src/approx/backpropagation/backpropagation.cpp -o bin/backpropagation -Iinclude -lstdc++fs && ./bin/backpropagation

// ./bin/generate_set 0.0 1.0 100000 "sin(t * 2.0 * 3.14) * 0.5 + 0.5" input/train.nse
// ./bin/generate_set 0.0 1.0 100 "sin(t * 2.0 * 3.14) * 0.5 + 0.5" input/test.nse
// ./bin/backpropagation 4 1 3 3 1 10.0 TanH input/teach.nse input/test.nse output/mc_network.neetwook

// g++ -O3 src/approx/backpropagation/backpropagation.cpp -o bin/backpropagation -Iinclude -lstdc++fs && ./bin/multistart_cutoff 4 1 3 3 1 10.0 TanH input/train.nse input/test.nse output/mc_network.neetwook

int main(int argc, char** argv) {
	
	// This example demonstrates performance testing of the backpropagation algorithm.
	// 
	// 1. Input data is being readed to regular training set.
	// 2. Network is being generated with generate_random with passed W value dispersion.
	// 3. Network is being teached on input data & measured time & amount of learning iterations.
	// 4. The results of the measuting is being printed out as regular.
	
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
	
	// -- Time record start here --
	auto timestamp_1 = std::chrono::high_resolution_clock::now();
	unsigned long learn_iterations_count = 0;
	
	// 2. Generate Networks
	NNSpace::MLNetwork network;
	NNSpace::generate_random_weight_network(network, dimensions, activator, Wd, 0, PRINT_BOOL);
		
	// 3. Learning

	// Collect statistics
	learn_iterations_count += train_set.size();
		
	// 3. Perform teaching of all networks
	NNSpace::train_network_backpropagation(network, train_set, 1, PRINT_BOOL);
	
	// -- Time record stop here --
	auto timestamp_2 = std::chrono::high_resolution_clock::now();

	// 10. Calculate resulting network error value & print
	long double error_value_result = NNSpace::calculate_linear_error(network, test_set, PRINT_BOOL);
	
	std::cout << "Result error value: " << error_value_result << std::endl;
	std::cout << "Result learning time: " << std::chrono::duration_cast<std::chrono::milliseconds>(timestamp_2 - timestamp_1).count() << "ms" << std::endl;
	std::cout << "Result train iterations: " << learn_iterations_count << std::endl;
	std::cout << "Serializing into: " << output_filename << std::endl;
	NNSpace::store_network(network, output_filename, PRINT_BOOL);
	
	return 0;
};