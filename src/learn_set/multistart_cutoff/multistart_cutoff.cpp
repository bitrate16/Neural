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

// g++ -O3 src/learn_set/multistart_cutoff/multistart_cutoff.cpp -o bin/multistart_cutoff -Iinclude -lstdc++fs

// ./bin/generate_set 0.0 1.0 100000 "sin(t * 2.0 * 3.14) * 0.5 + 0.5" input/train.nse
// ./bin/generate_set 0.0 1.0 100 "sin(t * 2.0 * 3.14) * 0.5 + 0.5" input/test.nse

// g++ -O3 src/learn_set/multistart_cutoff/multistart_cutoff.cpp -o bin/multistart_cutoff -Iinclude -lstdc++fs && ./bin/multistart_cutoff 3 784 100 10 10.0 1000 TanH input output/mc_network.neetwook

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
	// 3+L. test set slice size.
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
	
	/*
	// Input - image 28*28
	dimensions[0] = 28 * 28;
	// Output - vector of activations. Max is the answer.
	dimensions[L + 1] = 10;
	*/
	
	int A = 0;
	for (int i = 0; i < L - 1; ++i) 
		A += dimensions[i] * dimensions[i + 1];
	
	// Protect check to avoid system death on attempt to generate more than allowed.
#ifdef UPPER_WARNING_LIMIT
	if (A > UPPER_WARNING_LIMIT) {
		char c = 'a';
		std::cout << "Resulting networks size is too large (" << A << ")" << std::endl;
		
		while (c != 'y' && c != 'n' && c != 'Y' && c != 'N') {
			std::cout << "Proceed? [y/n]:";
			
			std::cin >> c;
		}
		
		if (c == 'n' || c == 'N') {
			std::cout << "Cancelling" << std::endl;
			return 0;
		}
	}
#endif
	
	// Weight distribution value
	double Wd = std::stod(argv[L + 2]);
	
	// Amount of test samples being used to test network error value
	int test_set_slice_size = std::stoi(argv[L + 3]);
	
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
	
	std::cout << "Training set size: " << dataset.training_images.size() << std::endl;
	std::cout << "Test set size: " << dataset.test_images.size() << std::endl;
	std::cout << "Error test set size: " << test_set_slice_size << std::endl;

	// -- Time record start here --
	auto timestamp_1 = std::chrono::high_resolution_clock::now();
	unsigned long learn_iterations_count = 0;
	
	// 3. Generate Networks
	std::vector<NNSpace::MLNetwork> networks;
	NNSpace::generate_random_weight_networks(networks, dimensions, activator, Wd, 0, PRINT_BOOL);
	
	std::vector<int> index_array;
	std::vector<int> half_index_array;
	
	std::vector<long double> V;
	std::vector<long double> E;
	
	long double E_min, V_max;
		
	// 4. Loop
	int step = 0;
	int N = NNSpace::log2(A);

	// Calculate iteration set size here
	// Ci = C1 * 2 ^ (i-1)
	// C1 = C / (2 ^ [log2(A)] - 1)
	int C1 = dataset.training_images.size() / ((1 << NNSpace::log2(A)) - 1);
	int Ci = C1;
	int current = 0;
	int current_size = C1;
	
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
		learn_iterations_count += networks.size() * current_size;
		
		for (int i = 0; i < networks.size(); ++i) {
			// 5. Calculate error value for each network now (V<b>)
			
			V[i] = NNSpace::calculate_mnist_linear_error(networks[i], dataset, test_set_slice_size, dimensions[L-1]);
			
			// 6. Perform teaching of all networks
			NNSpace::train_mnist_network_backpropagation(networks[i], dataset, current, current_size, dimensions[L-1], 1);
			
			// 8. Calculate Average error value now (E)
			E[i] = NNSpace::calculate_mnist_linear_error(networks[i], dataset, test_set_slice_size, dimensions[L-1]);
			if (E[i] < E_min)
				E_min = E[i];
			
			// 7. Calculate error value for each network now (V<a>)
			V[i] = E[i] - V[i];
			
			// Calculate V
			V[i] /= (long double) current_size;
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
		
		int Cj       = C1 << step + 1;
		current     += current_size;
		current_size = Cj;
		Ci           = Cj;
	}
	
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
	NNSpace::store_network(networks[0], output_filename, PRINT_BOOL);
	
	return 0;
};