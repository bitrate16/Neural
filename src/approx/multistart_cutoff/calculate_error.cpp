#include <iostream>
#include <string>
#include <cstring>

#include "MultistartCutoff.h"

#ifdef ENABLE_PRINT
	#define PRINT_BOOL 1
#else
	#define PRINT_BOOL 0
#endif

// g++ -O3 src/approx/multistart_cutoff/calculate_error.cpp -o bin/calculate_error -Iinclude -lstdc++fs && ./bin/calculate_error
// ./bin/generate_set 0.0 1.0 100 "sin(t * 3.14 * 2.0) * 0.5 + 0.5" input/test_set.nse
// ./bin/calculate_error 0 output/networks input/test_set.nse



int main(int argc, char** argv) {
	
	// calculate_error, args:
	// 0. id of network to calculate.
	// 1. path to network directory.
	// 2. path to test set.
	// Network is being dumped back to file where is was stored
	
	if (argc < 4) {
		std::cout << "Not enough arguments" << std::endl;
		return 0;
	}
	
	int net_id = std::stoi(argv[1]);
	std::string net_dir  = argv[2];
	std::string set_name = argv[3];
	
	std::vector<NNSpace::linear_set_point> set; 
	if (!NNSpace::read_linear_set(set, set_name, 0, PRINT_BOOL)) {
		std::cout << "Failed reading testing set" << std::endl;
		return 0;
	}
	
	NNSpace::MLNetwork network;
	if (!NNSpace::restore_network(network, net_dir + "/network_"+ std::to_string(net_id) + ".neetwook", PRINT_BOOL)) {
		std::cout << "Failed restore network" << std::endl;
		return 0;
	}
	
	// Calculate avg square error value
	long double mean = 0.0;
	std::vector<double> input(1);
	std::vector<double> output;
	for (int i = 0; i < set.size(); ++i) {
		input[0] = set[i].x;
		
		output = network.run(input);
		
		long double dv = output[0] - set[i].y;
		mean += dv * dv;
	}
	mean /= (long double) set.size();
	
#ifdef ENABLE_PRINT
	std::cout << "Error value: " << mean << std::endl;
#else
	std::cout << mean << std::endl;
#endif
	return 0;
};