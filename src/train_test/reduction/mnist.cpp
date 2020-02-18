#include <iostream>
#include <vector>
#include <chrono>
#include <limits>

#include "NetTestCommon.h"
#include "pargs.h"

/*
 * (c) Copyright bitrate16 (GPLv3.0) 2020
 * Performs simpla testing of passed amount of networks with passed aguments.
 * Testing on MNIST Digit recognition
 * Warning: Using Linear or ReLU activator may lead to double type overflow
 * Arguments:
 *  --network=%      Path to the network
 *  --mnist=%        Input set
 *  --test_size=%    Amount of digits taken from test set
 *  --test_offset=%  Offset value for test set
 *  --output=%       Output file for the network
 *  --Ltype=%        L1 or L2
 *  --error_dev=%    Max error deviation
 *  --print          Enable informational printing
 *  --log=[%]        Log type (REDUCTION_TIME, REDUCTION_ITERATIONS, TEST_ERROR_AVG, TEST_ERROR_MAX, TEST_MATCH, RECALC_ITERATIONS, NEURONS_REMOVED)
 *
 * Make:
 * g++ src/train_test/reduction/mnist.cpp -o bin/reduction_mnist -O3 --std=c++17 -Iinclude -lstdc++fs
 *
 * Example:
 * ./bin/reduction_mnist --test_size=1000 --print --mnist=data/mnist --network=networks/mnist_test.neetwook --output=networks/mnist_test_min.neetwook --log=[REDUCTION_TIME,TEST_ERROR_AVG,TEST_ERROR_MAX,REDUCTION_ITERATIONS,TEST_MATCH,RECALC_ITERATIONS,NEURONS_REMOVED]
 */

// Simply prints out the message and exits.
inline void exit_message(const std::string& message) {
	if (message.size())
		std::cout << message << std::endl;
	exit(0);
};

int main(int argc, const char** argv) {
	pargs::pargs args(argc, argv);
	
	// Read Ltype
	int Ltype = args["--Ltype"] ? args["--Ltype"]->get_integer() : 1;
	if (Ltype != 1 && Ltype != 2)
		Ltype = 1;
	
	// Read set
	std::string mnist_path = args["--mnist"] && args["--mnist"]->is_string() ? args["--mnist"]->string() : "mnist";
	
	// Read set
	mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> set;
	if (!NNSpace::Common::load_mnist(set, mnist_path))
		exit_message("Set " + mnist_path + " not found");
	
	// Parse limit properties
	int test_size = args["--test_size"]  ? args["--test_size"]->get_integer()  : -1;
	int test_offset = args["--test_offset"]  ? args["--test_offset"]->get_integer()  : 0;
	
	// Validate values
	if (test_size == -1)
		test_size = set.test_images.size();
	if (test_offset < 0 || test_size <= 0 || test_offset + test_size > set.test_images.size())
		exit_message("Invalid test offset or size");
	
	double error_dev = (args["--error_dev"] && args["--error_dev"]->is_real()) ? args["--error_dev"]->real() : 0.0;
	
	// Parse print flag
	bool print_flag = args["--print"];
	
	// Generate network
	NNSpace::MLNet network;
	if (!args["--network"])
		exit_message("No network specified");
	if (!NNSpace::Common::read_network(network, args["--network"]->string()))
		exit_message("Network " + args["--network"]->string() + " not found");;
	
	if (network.dimensions.size() <= 2)
		exit_message("Not enough deep layer size");
	
	// Perform testing
	auto start_time = std::chrono::high_resolution_clock::now();
	unsigned long reduction_iterations = 0;
	unsigned long recalculation_iterations = 0;
	unsigned long neurons_removed = 0;
	
	// Calculate initial value
	double initial_match = NNSpace::Common::calculate_mnist_match(network, set, test_offset, test_size);
	
	// Looping condition
	bool condition = 1;
	
	// Loop:
	while (condition) {
		++reduction_iterations;
		
		if (print_flag)
			std::cout << "Iteration: " << reduction_iterations << std::endl;
		
		// Check if dimensions are size of 1
		for (int i = 0; i < network.dimensions.size() - 2; ++i)
			if (network.dimensions[i + 1] == 1) { 
				condition = 0;
				break;
			}
		
		// Backup weights for each neuron
		std::vector<double> income;
		std::vector<double> outcome;
		double offset_store;
		
		// Minimal error value & neuron location
		double max_match = 0.0;
		int max_match_i = -1, max_match_j = -1;
		
		// Calculate Ri as match value without i neuron
		std::vector<std::vector<double>> R(network.dimensions.size() - 2);
		for (int i = 0; i < network.dimensions.size() - 2; ++i) {
			R[i].resize(network.dimensions[i + 1]);
			
			for (int j = 0; j < network.dimensions[i + 1]; ++j) {
				if (network.dimensions[i + 1] == 1)
					continue;
				
				++recalculation_iterations; 
				
				if (print_flag)
					std::cout << "Calculating R[" << (i + 1) << "][" << j << "], iteration: " << recalculation_iterations << std::endl;
				
				// Backup neuron weights, remove if from the network, calculate error, than return back
				income.resize(network.dimensions[i]);
				for (int a = 0; a < network.dimensions[i]; ++a)
					income[a] = network.W[i][a][j];
				
				outcome.resize(network.dimensions[i + 2]);
				for (int b = 0; b < network.dimensions[i + 2]; ++b)
					outcome[b] = network.W[i + 1][j][b];
				
				offset_store = network.offsets[i][j];
				
				// Erase
				--network.dimensions[i + 1];
				
				// Remove incoming
				for (int a = 0; a < network.dimensions[i]; ++a)
					network.W[i][a].erase(network.W[i][a].begin() + j);
				
				// Remove outcoming
				network.W[i + 1].erase(network.W[i + 1].begin() + j);
				
				// Remove offset
				network.offsets[i].erase(network.offsets[i].begin() + j);
				
				R[i][j] = NNSpace::Common::calculate_mnist_match(network, set, test_offset, test_size);
				
				// Record maximal match value
				if (max_match <= R[i][j]) {
					max_match = R[i][j];
					max_match_i = i;
					max_match_j = j;
				}
				
				// Restore
				++network.dimensions[i + 1];
				
				// Insert incoming
				for (int a = 0; a < network.dimensions[i]; ++a)
					network.W[i][a].insert(network.W[i][a].begin() + j, income[a]);
				
				// Insert outcoming
				network.W[i + 1].insert(network.W[i + 1].begin() + j, outcome);
				
				// Insert offset
				network.offsets[i].insert(network.offsets[i].begin() + j, offset_store);
			}
		}
		
		if (print_flag) {
			std::cout << "Initial match = " << initial_match << std::endl;
			std::cout << "Iteration maximal match = " << max_match << std::endl;
		}
		
		// Remove neuron[i][j] if (initial - R[i][j]) < error_dev
		if (initial_match - max_match <= error_dev) {
			if (print_flag)
				std::cout << "Iteration match += " << (max_match - initial_match) << std::endl;
			
			// Erase
			--network.dimensions[max_match_i + 1];
			
			// Remove incoming
			for (int a = 0; a < network.dimensions[max_match_i]; ++a)
				network.W[max_match_i][a].erase(network.W[max_match_i][a].begin() + max_match_j);
			
			// Remove outcoming
			network.W[max_match_i + 1].erase(network.W[max_match_i + 1].begin() + max_match_j);
			
			// Remove offset
			network.offsets[max_match_i].erase(network.offsets[max_match_i].begin() + max_match_j);
			
			++neurons_removed;
			
			if (print_flag)
				std::cout << "Iteration new layer [" << (max_match_i + 1) << "] size: " << network.dimensions[max_match_i + 1] << std::endl;
		} else 
			condition = false;
	}
	
	auto end_time = std::chrono::high_resolution_clock::now();
	
	// Do logging of the requested values
	if (args["--log"]) {
		if (args["--log"]->array_contains("REDUCTION_TIME")) {
			
			// Calculate time used
			auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
			std::cout << "REDUCTION_TIME=" << train_time << "ms" << std::endl;
		}
		if (args["--log"]->array_contains("RECALC_ITERATIONS"))
			std::cout << "RECALC_ITERATIONS=" << recalculation_iterations << std::endl;
		if (args["--log"]->array_contains("REDUCTION_ITERATIONS"))
			std::cout << "REDUCTION_ITERATIONS=" << reduction_iterations << std::endl;
		if (args["--log"]->array_contains("TEST_MATCH")) 
			std::cout << "TEST_MATCH=" << NNSpace::Common::calculate_mnist_match(network, set, test_offset, test_size) << std::endl;
		if (args["--log"]->array_contains("TEST_ERROR_AVG")) 
			std::cout << "TEST_ERROR_AVG=" << NNSpace::Common::calculate_mnist_error(network, set, Ltype, test_offset, test_size) << std::endl;
		if (args["--log"]->array_contains("TEST_ERROR_MAX")) 
			std::cout << "TEST_ERROR_MAX=" << NNSpace::Common::calculate_mnist_error_max(network, set, Ltype, test_offset, test_size) << std::endl;
		if (args["--log"]->array_contains("NEURONS_REMOVED")) 
			std::cout << "NEURONS_REMOVED=" << neurons_removed << std::endl;
	}
	
	// Write network to file
	if (args["--output"] && args["--output"]->is_string())
		NNSpace::Common::write_network(network, args["--output"]->string());
	
	return 0;
};