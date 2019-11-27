#include <iostream>
#include <string>
#include <cstring>

#include "MultistartCutoff.h"

#ifdef ENABLE_PRINT
	#define PRINT_BOOL 1
#else
	#define PRINT_BOOL 0
#endif

// g++ -O3 src/multistart_cutoff/generate_networks.cpp -o bin/generate_networks -Iinclude -lstdc++fs && ./bin/generate_networks
// ./bin/generate_networks 4 1 3 3 1 TanH output/networks


int main(int argc, char** argv) {
	
	// generate_networks, args:
	// 0. number of layers ([G.d]).
	// 1-L+1. layer dimensions (G.l[k = 1, G.d]).
	// L+2. Activator function type T (TanH, Sigmoid, Linear).
	// L+3. Output directory.
	
	if (argc < 2) {
		std::cout << "Not enough arguments" << std::endl;
		return 0;
	}
	
	int layers = std::stoi(argv[1]);
	if (layers < 3) {
		std::cout << "Layers count < 4" << std::endl;
		return 0;
	}
	
	if (argc < 2 + layers + 1 + 1) {
		std::cout << "Not enough arguments" << std::endl;
		return 0;
	}
	
	std::vector<int> dimensions(layers);
	for (int i = 0; i < layers; ++i)
		dimensions[i] = std::stoi(argv[i + 2]);
	
	NNSpace::ActivatorType activator;
	if (strcmp("TanH", argv[layers + 2]) == 0)
		activator = NNSpace::ActivatorType::TANH;
	else if (strcmp("Sigmoid", argv[layers + 2]) == 0)
		activator = NNSpace::ActivatorType::SIGMOID;
	else if (strcmp("Linear", argv[layers + 2]) == 0)
		activator = NNSpace::ActivatorType::LINEAR;
	else {
		std::cout << "Invalid activator type" << std::endl;
		return 0;
	}
	
	std::string output_dir = argv[layers + 3];
	
	// Generate set of networks with specified parameters using 
	//  > generate_networks
	
	NNSpace::remove_directory(output_dir, PRINT_BOOL);
	if (!NNSpace::generate_random_weight_networks(dimensions, output_dir, activator, 0, PRINT_BOOL)) {
		std::cout << "Failed creating networks" << std::endl;
		return 0;
	}
	
	return 0;
};