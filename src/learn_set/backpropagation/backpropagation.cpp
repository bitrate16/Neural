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

// Limit of networks size after which the warning should be printed out
#define UPPER_WARNING_LIMIT 10000

// #define DEBUG_CUT_SET

// g++ -O3 src/learn_set/backpropagation/backpropagation.cpp -o bin/backpropagation -Iinclude -lstdc++fs

// g++ -O3 src/learn_set/backpropagation/backpropagation.cpp -o bin/backpropagation -Iinclude -lstdc++fs && ./bin/backpropagation 3 784 100 10 10.0 TanH input output/mc_network.neetwook


int main(int argc, char** argv) {
	
	// The source for this algorithm is based on < - - insert diploma here - - > [1]
	
	// This method is combined from standard backpropagation and 
	//  cutting off the unperspective starts.
	
	// This example does multistart cutoff algorithm.
	// Example of teaching neural network on MNIST dataset.
	
	// Input:
	// 1. layers count [L].
	// >   layer [0] size is assumed to be input size of dataset
	// >   layer[L-1] size is assumed to be output size of dataset
	// 2+i. layer i size.
	// 2+L. weights dispersion.
	// 3+L. activator function (TanH, Sigmoid, Linear).
	// 4+L. MNIST dataset location (folder).
	// 5+L. output file for network
	
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
	
	for (int i = 0; i < L; ++i)
		dimensions[i] = std::stoi(argv[i + 2]);
	
	// Weight distribution value
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
	
	std::string set_file = argv[L + 4];
	std::string output_filename = argv[L + 5];
	
	// 1. Read train & test set	
	mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(set_file);
	
	std::cout << "Training set size: " << dataset.training_images.size() << std::endl;
	std::cout << "Test set size: " << dataset.test_images.size() << std::endl;

	// -- Time record start here --
	auto timestamp_1 = std::chrono::high_resolution_clock::now();
	unsigned long learn_iterations_count = 0;
	
	// 3. Generate Networks
	NNSpace::MLNetwork network;
	NNSpace::generate_random_weight_network(network, dimensions, activator, Wd, 0, PRINT_BOOL);
	
	// Collect statistics 
	learn_iterations_count += dataset.training_images.size();
	
	// 5. Perform teaching of network
	NNSpace::train_mnist_network_backpropagation(network, dataset, 0, dataset.training_images.size(), dimensions[L-1], 1);
	
	// -- Time record stop here --
	auto timestamp_2 = std::chrono::high_resolution_clock::now();

	// 10. Calculate resulting network error value & print
	long double error_value_result = NNSpace::calculate_mnist_match_error(network, dataset, dataset.test_images.size(), dimensions[L-1]);
	
	std::cout << "Result error value: " << error_value_result << " [" << ((1.0 - error_value_result) * dataset.test_images.size()) << " / " << dataset.test_images.size() << "] ";
	std::cout.precision(4);
	std::cout << ((1.0 - error_value_result) * 100) << "%" << std::endl;
	std::cout << "Result learning time: " << std::chrono::duration_cast<std::chrono::milliseconds>(timestamp_2 - timestamp_1).count() << "ms" << std::endl;
	std::cout << "Result train iterations: " << learn_iterations_count << std::endl;
	std::cout << "Serializing into: " << output_filename << std::endl;
	NNSpace::store_network(network, output_filename, PRINT_BOOL);
	
	return 0;
};