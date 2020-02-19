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
 *  --test=%         Input set
 *  --output=%       Output file for the network
 *  --Ltype=%        L1 or L2
 *  --error_dev=%    Max error deviation
 *  --print          Enable informational printing
 *  --log=[%]        Log type (REDUCTION_TIME, REDUCTION_ITERATIONS, TEST_ERROR_AVG, TEST_ERROR_MAX, RECALC_ITERATIONS, NEURONS_REMOVED)
 *
 * Make:
 * g++ src/train_test/reduction/approx_2d.cpp -o bin/reduction_approx_2d -O3 --std=c++17 -Iinclude -lstdc++fs
 *
 * Example:
 * ./bin/reduction_approx_2d --print --error_dev=0.1 --test=data/sin_1000.mset --network=networks/approx_sin.neetwook --output=networks/approx_sin_min.neetwook --log=[REDUCTION_TIME,TEST_ERROR_AVG,TEST_ERROR_MAX,REDUCTION_ITERATIONS,TEST_MATCH,RECALC_ITERATIONS,NEURONS_REMOVED]
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
	
	// Reap input
	std::string test = args["--test"] && args["--test"]->is_string() ? args["--test"]->string() : "test.mset";
	
	// Read train set data
	std::vector<std::pair<double, double>> test_set;
	if (!NNSpace::Common::read_approx_set(test_set, test))
		exit_message("Set " + test + " not found");
	
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
	double initial_error = NNSpace::Common::calculate_approx_error(network, test_set, Ltype);
	
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
		double min_error = std::numeric_limits<double>::max();
		int min_error_i = -1, min_error_j = -1;
		
		// Calculate Ri as error value without i neuron
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
				
				R[i][j] = NNSpace::Common::calculate_approx_error(network, test_set, Ltype);
				
				// Record maximal match value
				if (min_error >= R[i][j]) {
					min_error = R[i][j];
					min_error_i = i;
					min_error_j = j;
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
			std::cout << "Initial error = " << initial_error << std::endl;
			std::cout << "Iteration minimal error = " << min_error << std::endl;
		}
		
		// Remove neuron[i][j] if (initial - R[i][j]) < error_dev
		if (min_error - initial_error <= error_dev) {
			if (print_flag)
				std::cout << "Iteration error += " << (initial_error - min_error) << std::endl;
			
			// Erase
			--network.dimensions[min_error_i + 1];
			
			// Remove incoming
			for (int a = 0; a < network.dimensions[min_error_i]; ++a)
				network.W[min_error_i][a].erase(network.W[min_error_i][a].begin() + min_error_j);
			
			// Remove outcoming
			network.W[min_error_i + 1].erase(network.W[min_error_i + 1].begin() + min_error_j);
			
			// Remove offset
			network.offsets[min_error_i].erase(network.offsets[min_error_i].begin() + min_error_j);
			
			++neurons_removed;
			
			if (print_flag)
				std::cout << "Iteration new layer [" << (min_error_i + 1) << "] size: " << network.dimensions[min_error_i + 1] << std::endl;
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
		if (args["--log"]->array_contains("TEST_ERROR_AVG")) 
			std::cout << "TEST_ERROR_AVG=" << NNSpace::Common::calculate_approx_error(network, test_set, Ltype) << std::endl;
		if (args["--log"]->array_contains("TEST_ERROR_MAX")) 
			std::cout << "TEST_ERROR_MAX=" << NNSpace::Common::calculate_approx_error_max(network, test_set, Ltype) << std::endl;
		if (args["--log"]->array_contains("NEURONS_REMOVED")) 
			std::cout << "NEURONS_REMOVED=" << neurons_removed << std::endl;
	}
	
	// Write network to file
	if (args["--output"] && args["--output"]->is_string())
		NNSpace::Common::write_network(network, args["--output"]->string());
	
	return 0;
};