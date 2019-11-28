#include <iostream>
#include <string>
#include <cstring>

#include "NetworkTestingCommon.h"

#ifdef ENABLE_PRINT
	#define PRINT_BOOL 1
#else
	#define PRINT_BOOL 0
#endif

// g++ -O3 src/approx/split_set.cpp -o bin/split_set -Iinclude -lstdc++fs && ./bin/split_set
// ./bin/split_set input/train_set.nse input/train_set 10



int main(int argc, char** argv) {
	
	// split_set, args:\
	// 1. path to train set.
	// 2. path to split train set directory.
	// 3. split count
	
	if (argc < 4) {
		std::cout << "Not enough arguments" << std::endl;
		return 0;
	}
	
	std::string in_name = argv[1];
	std::string out_dir = argv[2];
	int splits = std::stoi(argv[3]);
	
	NNSpace::remove_directory(out_dir, PRINT_BOOL);
	if (!NNSpace::split_linear_set(in_name, out_dir, splits, 0, PRINT_BOOL)) {
		std::cout << "Failed split set" << std::endl;
		return 0;
	}
	
	return 0;
};