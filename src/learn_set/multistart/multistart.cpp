#include <iostream>
#include <string>
#include <limits>
#include <cstring>
#include <cmath>
#include <chrono>
#include <csignal>

#include "NetworkTestingCommon.h"

#ifdef ENABLE_PRINT
	#define PRINT_BOOL 1
#else
	#define PRINT_BOOL 0
#endif

// Limit of networks size after which the warning should be printed out
#define UPPER_WARNING_LIMIT 10000

// #define DEBUG_CUT_SET

// g++ -O3 src/learn_set/multistart/multistart.cpp -o bin/multistart -Iinclude -lstdc++fs

// g++ -O3 src/learn_set/multistart/multistart.cpp -o bin/multistart -Iinclude -lstdc++fs && ./bin/multistart 3 784 100 10 10.0 100 TanH input output/mc_network.neetwook

// Allow interrupt learning process and dump current results
bool learning_state = 1;

void interrupt_signal(int signum) {
	signal(SIGINT, interrupt_signal);
	learning_state = 0;
}


int main(int argc, char** argv) {
	
	// This method is a standard backpropagation with multiple start networks.
	// Example of teaching neural network on MNIST dataset.
	
	// Input:
	// 1. layers count [L].
	// >   layer [0] size is assumed to be input size of dataset
	// >   layer[L-1] size is assumed to be output size of dataset
	// 2+i. layer i size.
	// 2+L. weights dispersion.
	// 3+L. start networks count.
	// 4+L. activator function (TanH, Sigmoid, Linear).
	// 5+L. MNIST dataset location (folder).
	// 6+L. output file for network.
	
	// Output: error value for test set on produced network.
	
	
	// 0. Parse input
	if (argc < 2) {
		std::cout << "Not enough arguments" << std::endl;
		return 0;
	}
	
	int L = std::stoi(argv[1]);
	
	if (argc < L + 7)  {
		std::cout << "Not enough arguments" << std::endl;
		return 0;
	}
	
	std::vector<int> dimensions(L);
	
	for (int i = 0; i < L; ++i)
		dimensions[i] = std::stoi(argv[i + 2]);
	
	// Weight distribution value
	double Wd = std::stod(argv[L + 2]);
	
	// Amount of test samples being used to test network error value
	int networks_count = std::stoi(argv[L + 3]);
	
	NNSpace::ActivatorType activator;
	if (strcmp("TanH", argv[L + 4]) == 0)
		activator = NNSpace::ActivatorType::TANH;
	else if (strcmp("Sigmoid", argv[L + 4]) == 0)
		activator = NNSpace::ActivatorType::SIGMOID;
	else if (strcmp("Linear", argv[L + 4]) == 0)
		activator = NNSpace::ActivatorType::LINEAR;
	else {
		std::cout << "Invalid activator type" << std::endl;
		return 0;
	}
	
	std::string set_file = argv[L + 5];
	std::string output_filename = argv[L + 6];
	
	
	// 1. Read train & test set	
	mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(set_file);
	
	std::cout << "Training set size:   " << dataset.training_images.size() << std::endl;
	std::cout << "Test set size:       " << dataset.test_images.size() << std::endl;

	// -- Time record start here --
	auto timestamp_1 = std::chrono::high_resolution_clock::now();
	unsigned long learn_iterations_count = 0;
	
	// 3. Generate Networks
	std::vector<NNSpace::MLNetwork> networks;
	NNSpace::generate_random_weight_networks_count(networks, dimensions, activator, Wd, networks_count, 0, PRINT_BOOL);
	
	int         min_error_id = 0;
	long double min_error    = std::numeric_limits<long double>::max();
	
	// Set up interrupt listener
	signal(SIGINT, interrupt_signal);
	
	// 4. Train networks
	for (int i = 0; i < networks_count && learning_state; ++i) {
		
		NNSpace::train_mnist_network_backpropagation(networks[i], dataset, 0, dataset.training_images.size(), dimensions[L-1], 1);
		
		long double error = NNSpace::calculate_mnist_match_error(networks[i], dataset, dataset.test_images.size(), dimensions[L-1]);
		std::cout << "Network #" << i << " error: " << error << std::endl;
		
		learn_iterations_count += dataset.training_images.size();
		
		if (error < min_error) {
			min_error = error;
			min_error_id = i;
		}
	}
	
	// -- Time record stop here --
	auto timestamp_2 = std::chrono::high_resolution_clock::now();
	
	std::cout << "Result error value: " << min_error << " [" << ((1.0 - min_error) * dataset.test_images.size()) << " / " << dataset.test_images.size() << "] ";
	std::cout.precision(4);
	std::cout << ((1.0 - min_error) * 100) << "%" << std::endl;
	std::cout << "Result learning time: " << std::chrono::duration_cast<std::chrono::milliseconds>(timestamp_2 - timestamp_1).count() << "ms" << std::endl;
	std::cout << "Result train iterations: " << learn_iterations_count << std::endl;
	std::cout << "Serializing into: " << output_filename << std::endl;
	NNSpace::store_network(networks[min_error_id], output_filename, PRINT_BOOL);
	
	return 0;
};