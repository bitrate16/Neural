#include <iostream>
#include <string>

#include "NetworkTestingCommon.h"

int main() {
	
	// > generate_set.cpp      - generates distributed over line and randomly shuffle
	// > generate_networks.cpp - random networks generator
	// > train_network.cpp     - trains selected network
	// > split_set.cpp         - splits given set into several subsets
	// > calculate_mean.cpp    - calculate error value on passed network 
	
	// input - Graph Topo (G, T)
	// G - Oriented graph (G.l - leyars, G.d - number of layers)
	// T - Activator functions set.
	
	// multistart_test, args:
	// 0. number of layers ([G.d]).
	// 1-L+1. layer dimensions (G.l[k = 1, G.d]).
	// L+2. Activator function type T (TanH, Sigmoid, Linear).
	// L+3. learn set path (Lp).
	// L+4. test set path (Tp).
	
	if (argc < 2) {
		std::cout << "Not enough arguments" << std::endl;
		return 0;
	}
	
	int 
	
	// Generate set of networks with specified parameters using 
	//  > generate_networks
	
	
};