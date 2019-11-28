#include <iostream>
#include <string>
#include <cstring>

#include "NetworkTestingCommon.h"

#ifdef ENABLE_PRINT
	#define PRINT_BOOL 1
#else
	#define PRINT_BOOL 0
#endif

// g++ -O3 src/approx/multistart_cutoff/train_network.cpp -o bin/train_network -Iinclude -lstdc++fs && ./bin/train_network
// ./bin/generate_set 0.0 1.0 1000 "sin(t * 3.14 * 2.0) * 0.5 + 0.5" input/train_set.nse
// ./bin/train_network 0 output/networks input/train_set.mse



int main(int argc, char** argv) {
	
	// train_network, args:
	// 0. id of network to train.
	// 1. path to network directory.
	// 2. path to train set.
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
		std::cout << "Failed reading train set" << std::endl;
		return 0;
	}
	
	NNSpace::MLNetwork network;
	if (!NNSpace::restore_network(network, net_dir + "/network_"+ std::to_string(net_id) + ".neetwook", PRINT_BOOL)) {
		std::cout << "Failed restore network" << std::endl;
		return 0;
	}
	
	NNSpace::train_network_backpropagation(network, set);
	
	if (!NNSpace::store_network(network, net_dir + "/network_"+ std::to_string(net_id) + ".neetwook", PRINT_BOOL)) {
		std::cout << "Failed store network" << std::endl;
		return 0;
	}
	
	return 0;
};