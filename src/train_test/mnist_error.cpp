#include <iostream>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <limits>

#include "NetTestCommon.h"
#include "pargs.h"

/*
 * Make:
 * g++ src/train_test/mnist_error.cpp -o bin/mnist_error -O3 --std=c++17 -Iinclude -lstdc++fs
 * 
 * Run:
 * ./bin/mnist_error --mnist=data/mnist --test_size=100 --network=networks/mnist_test.neetwook 
 */

// Simply prints out the message and exits.
inline void exit_message(const std::string& message) {
	if (message.size())
		std::cout << message << std::endl;
	exit(0);
};

int main(int argc, const char** argv) {
	pargs::pargs args(argc, argv);
	NNSpace::MLNet network;
	NNSpace::Common::read_network(network, args["--network"]->string());
	
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
	int test_size  = args["--test_size"]  ? args["--test_size"]->get_integer()  : -1;
	int test_offset  = args["--test_offset"]  ? args["--test_offset"]->get_integer()  : 0;
	
	int steps = args["--steps"] ? args["--steps"]->get_integer() : 1;
	
	if (test_size == -1)
		test_size = set.test_images.size();
	if (test_offset < 0 || test_size <= 0 || test_offset + test_size > set.test_images.size())
		exit_message("Invalid test offset or size");
	
	std::cout << "TEST_MATCH=" << NNSpace::Common::calculate_mnist_match(network, set, test_offset, test_size) << std::endl;
	std::cout << "TEST_ERROR_AVG=" << NNSpace::Common::calculate_mnist_error(network, set, Ltype, test_offset, test_size) << std::endl;
	std::cout << "TEST_ERROR_MAX=" << NNSpace::Common::calculate_mnist_error_max(network, set, Ltype, test_offset, test_size) << std::endl;
	
	return 0;
};