#include <iostream>
#include <string>
#include <cstring>

#include "MultistartCutoff.h"

#ifdef ENABLE_PRINT
	#define PRINT_BOOL 1
#else
	#define PRINT_BOOL 0
#endif

// g++ -O3 src/approx/multistart_cutoff/multistart_cutoff.cpp -o bin/multistart_cutoff -Iinclude -lstdc++fs && ./bin/multistart_cutoff

// ./bin/generate_set 0.0 1.0 10000 "sin(t * 2.0 * 3.14) * 0.5 + 0.5" input/train.nse
// ./bin/generate_set 0.0 1.0 100 "sin(t * 2.0 * 3.14) * 0.5 + 0.5" input/test.nse
// ./bin/multistart_cutoff 4 1 3 3 1 TanH input/teach.nse input/test.nse

int main(int argc, char** argv) {
	
	// The source for this algorithm is based on < - - insert diploma here - - > [1]
	
	// This example does multistart cutoff algorighm.
	// Example of teaching neural network to approximate function Y = F(X)
	//  Where X, Y are scalar.
	// 1. Generating set of networks (generate_random_weight_networks).
	// > Refering to [1], total amount of networks in the networks set is A = W * n,
	// >  where W - total amount of weights, n - total amount of input neuroms.
	// > Proctically, than more neurons network has, than more variations of weights may 
	// >  occur, and than more local minimas must be excluded during the training.
	// > In this case the primitive algorighm is used:
	// >  Each network in set is being initialized with random set of weights distributed on [0.5, 0.5].
	// >  Next update step is to implement k-means method to split weights into k 
	// >   disjoint point classes and generate weights for each class.
	// 
	// 2. Read train set & split into set of subsets (split_linear_set) with sizes of Ci.
	// > Primitive set used in comparison between standart backpropagation and M-C.
	// >  The task of the network is to approximate the given function and 
	// >   minimize the error value for given topology.
	// >  Basing on current task, the methosics of set generation is splitting 
	// >   function describe area on line into R sectors, combine (X, Y) pairs 
	// >   and randomly shuffle. Such method is used to generate common 
	// >   test / teach set for both teach algorithms.
	// 
	// 3. For all left sets on step i perform Ci steps of learning 
	//     (learn network on Ci examples) using local-optimal algorithm.
	//     Current implementation is backpropagation algorithm.
	// > Referring to [1], total amount of steps in the algorithm is N = [log2(A)].
	// > Each step network is being reduced by Ai / 2, where Ai - amount of starts, 
	// >  left from the previous step.
	// > Each step amount of learn iterations is inversely to the amount of the left networks.
	// > Ai = [A / 2^i] is the number of networks left on i step.
	// > C1 = C / (2 ^ [log2(A)] - 1) is the size of learning set on 1st step.
	// > Ci = C1 * 2^(i-1) is the size of learning set on i step. Ci must grow on i growing.
	// > Referring to [1], to reduce amount of overlearning the network, Ci -= Ci / [log2(A)].
	// > Referring to [1], next update is to implement network set grouping 
	// >  and levenberg-marquardt learning algorithm.
	// 
	// 4. For all networks calculate gradient of error value V, average error value on set E.
	// > V is being calculated as average square error value difference of 
	// >  error values before teach and after teach.
	// >  V<b>i = SUM [ output<b> - test ]i ^ 2 / SET_SIZE
	// >  V<a>i = SUM [ output<a> - test ]i ^ 2 / SET_SIZE
	// >  V = V<b> - V<a>
	// > E is calculated as average linear error value during test
	// 
	// 5. Sort all networks ascending by V.
	//    Sort all networks descending by E.
	// > As the result we get a set of networks with Maximal E and minimal V.
	// 
	// 6. Select Ai networks and remove them from set.
	// 
	// 7. Repeat until N.
	//
	// 8. Left network is the resulting network that has mininal 
	//     error value for current topology.
	
	// > generate_set.cpp      - generates distributed over line and randomly shuffle
	// > generate_networks.cpp - random networks generator
	// > train_network.cpp     - trains selected network
	// > split_set.cpp         - splits given set into several subsets
	// > calculate_mean.cpp    - calculate error value on passed network 
	
	// Input:
	// 1. layers count [L].
	// 2+i. layer i size.
	// 2+L. activator function (TanH, Sigmoid, Linear).
	// 3+L. input train set.
	// 4+L. input test set.
	
	// Output: error value for test set on produced network.
	
	
	// 0. Parse input
	if (argc < 2) {
		std::cout << "Not enough arguments" << std::endl;
		return 0;
	}
	
	int L = std::stoi(argv[1]);
	
	if (argc < L + 5)  {
		std::cout << "Not enough arguments" << std::endl;
		return 0;
	}
	
	std::vector<int> dimensions(L);
	
	for (int i = 0; i < dimensions.size(); ++i)
		dimensions[i] = std::stoi(argv[i + 2]);
	
	NNSpace::ActivatorType activator;
	if (strcmp("TanH", argv[L + 2]) == 0)
		activator = NNSpace::ActivatorType::TANH;
	else if (strcmp("Sigmoid", argv[L + 2]) == 0)
		activator = NNSpace::ActivatorType::SIGMOID;
	else if (strcmp("Linear", argv[L + 2]) == 0)
		activator = NNSpace::ActivatorType::LINEAR;
	else {
		std::cout << "Invalid activator type" << std::endl;
		return 0;
	}
	
	std::string teach_set_file = argv[L + 3];
	std::string train_set_file = argv[L + 4];
	
	
	// 1. Read train & test set
	std::vector<NNSpace::linear_set_point> train_set;
	if (!NNSpace::read_linear_set(train_set, train_set_file, 0, PRINT_BOOL)) {
		std::cout << "Failed reading train set" << std::endl;
		return 0;
	}
	
	std::vector<NNSpace::linear_set_point> test_set;
	if (!NNSpace::read_linear_set(test_set, train_set_file, 0, PRINT_BOOL)) {
		std::cout << "Failed reading test set" << std::endl;
		return 0;
	}
	
	// 2. Generate train subsets
	int A = 0;
	for (int i = 0; i < L - 1; ++i) 
		A += dimensions[i] * dimensions[i + 1];
	
	std::vector<std::vector<NNSpace::linear_set_point>> train_set_set;
	NNSpace::split_linear_set_base_2(train_set_set, train_set, A, 0, PRINT_BOOL);
	
	// -- Time record start here --
	
	// 3. Generate Networks
	std::vector<NNSpace::MLNetwork> networks;
	generate_random_weight_networks(networks, dimensions, activator, 0, PRINT_BOOL);
	
	// 4. Loop
	int step = 0;
	int N = NNSpace::log2(a);
	while (step < N) {
		// 5. Calculate error value for each network now (V<b>)
		
		// 6. Perform teaching of all networks
		
		// 7. Calculate error value for each network now (V<a>)
		
		// 8. Calculate Average error value now (E)
		
		// 8. Order networks by error gradient (descending)
		
		// 9. Order networks by average error value (ascending)
		
		// 10. Select Ai networks and remove
		
	}
	
	// 11. Calculate resulting network error value & print
	
	// -- Time record stop here --
	
	return 0;
};