#include <iostream>
#include <vector>
#include <chrono>

#include "train/backpropagation.h"
#include "NetTestCommon.h"
#include "pargs.h"

/*
 * (c) Copyright bitrate16 (GPLv3.0) 2020
 * Performs simpla testing of passed amount of networks with passed aguments.
 * Testing on MNIST Digit recognition
 * Warning: Using Linear or ReLU activator may lead to double type overflow
 * Arguments:
 *  --layers=[%]     layer sizes
 *                   Not including the input, output layers. They are 1, 1.
 *  --activator=[%]  Activator[i] function type
 *  --weight=%       Weight dispersion
 *  --offsets=%      Enable offfsets flag
 *  --mnist=%        Input set
 *  --train_size=%   Amount of digits taken from train set
 *  --test_size=%    Amount of digits taken from test set
 *  --train_offset=% Offset value for train set
 *  --test_offset=%  Offset value for test set
 *  --output=%       Output file for the network
 *  --rate_factor=%  Scale factor for rate value
 *  --Ltype=%        L1 or L2
 *  --log=[%]        Log type (TRAIN_TIME, TRAIN_OPERATIONS, TRAIN_ITERATIONS, TEST_ERROR_AVG, TEST_ERROR_MAX, TEST_MATCH)
 *
 * Make:
 * g++ src/train_test/backpropagation/mnist.cpp -o bin/backpropagation_mnist -O3 --std=c++17 -Iinclude -lstdc++fs
 *
 * Example:
 * ./bin/backpropagation_mnist --layers=[3] --train_size=10000 --test_size=100 --offsets=true --activator=TanH --rate_factor=0.5 --weight=1.0 --mnist=data/mnist --output=networks/mnist_test.neetwook --log=[TRAIN_TIME,TEST_ERROR_AVG,TEST_ERROR_MAX,TRAIN_ITERATIONS,TEST_MATCH]
 * 
 * 0.817:
 * ./bin/backpropagation_mnist --layers=[] --train_size=10000 --test_size=1000 --offsets=true --activator=Sigmoid --rate_factor=1.0 --weight=1.0 --mnist=data/mnist --output=networks/mnist_test.neetwook --log=[TRAIN_TIME,TEST_ERROR_AVG,TEST_ERROR_MAX,TRAIN_ITERATIONS,TEST_MATCH]
 */

// Simply prints out the message and exits.
inline void exit_message(const std::string& message) {
	if (message.size())
		std::cout << message << std::endl;
	exit(0);
};

int main(int argc, const char** argv) {
	pargs::pargs args(argc, argv);
	
	// Read input data for network definition
	std::vector<int> dimensions = { 28 * 28, 10 };
	
	// Fill with layer dimensions
	if (args["--layers"] && args["--layers"]->is_array()) {
		dimensions.resize(args["--layers"]->array().size() + 2);
		
		for (int i = 0; i < args["--layers"]->array().size(); ++i) {
			dimensions[i + 1] = args["--layers"]->array()[i]->integer();
			
			if (!dimensions[i + 1])
				exit_message("Zero layer size");
		}
	}
	dimensions.back() = 10;
	
	// Read weight info
	//  Input or 1.0
	double wD = args["--weight"] && args["--weight"]->is_real() ? args["--weight"]->real() : 1.0;
	
	// Parse rate factor
	double rate_factor = args["--rate_factor"] && args["--rate_factor"]->is_real() ? args["--rate_factor"]->real() : 1.0;
	
	// Read offsets flag
	bool offsets = args["--offsets"] && args["--offsets"]->get_boolean();
	
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
	int train_size = args["--train_size"] ? args["--train_size"]->get_integer() : -1;
	int test_size  = args["--test_size"]  ? args["--test_size"]->get_integer()  : -1;
	int train_offset = args["--train_offset"] ? args["--train_offset"]->get_integer() : 0;
	int test_offset  = args["--test_offset"]  ? args["--test_offset"]->get_integer()  : 0;
	
	// Validate values
	if (train_size == -1)
		train_size = set.training_images.size();
	if (train_offset < 0 || train_size <= 0 || train_offset + train_size > set.training_images.size())
		exit_message("Invalid train offset or size");
	
	if (test_size == -1)
		test_size = set.test_images.size();
	if (test_offset < 0 || test_size <= 0 || test_offset + test_size > set.test_images.size())
		exit_message("Invalid test offset or size");
	
	// Generate network
	NNSpace::MLNet network;
	NNSpace::Common::generate_random_network(network, dimensions, wD, offsets);
	
	// Add activators (default is linear)
	if (args["--activator"]) {
		if (args["--activator"]->is_string()) {
			if (args["--activator"]->string() == "Linear")          network.setActivator(new NNSpace::Linear()        );
			if (args["--activator"]->string() == "Sigmoid")         network.setActivator(new NNSpace::Sigmoid()       );
			if (args["--activator"]->string() == "BipolarSigmoid")  network.setActivator(new NNSpace::BipolarSigmoid());
			if (args["--activator"]->string() == "ReLU")            network.setActivator(new NNSpace::ReLU()          );
			if (args["--activator"]->string() == "TanH")            network.setActivator(new NNSpace::TanH()          );
		} else if (args["--activator"]->is_integer()) {
			if (args["--activator"]->integer() == NNSpace::ActivatorType::LINEAR)          network.setActivator(new NNSpace::Linear()        );
			if (args["--activator"]->integer() == NNSpace::ActivatorType::SIGMOID)         network.setActivator(new NNSpace::Sigmoid()       );
			if (args["--activator"]->integer() == NNSpace::ActivatorType::BIPOLAR_SIGMOID) network.setActivator(new NNSpace::BipolarSigmoid());
			if (args["--activator"]->integer() == NNSpace::ActivatorType::RELU)            network.setActivator(new NNSpace::ReLU()          );
			if (args["--activator"]->integer() == NNSpace::ActivatorType::TANH)            network.setActivator(new NNSpace::TanH()          );
		} else if (args["--activator"]->is_array()) {
			for (int i = 0; i < args["--activator"]->array().size(); ++i) {
				if (args["--activator"]->array()[i]->is_string()) {
					if (args["--activator"]->array()[i]->string() == "Linear")          network.setActivator(new NNSpace::Linear()        );
					if (args["--activator"]->array()[i]->string() == "Sigmoid")         network.setActivator(new NNSpace::Sigmoid()       );
					if (args["--activator"]->array()[i]->string() == "BipolarSigmoid")  network.setActivator(new NNSpace::BipolarSigmoid());
					if (args["--activator"]->array()[i]->string() == "ReLU")            network.setActivator(new NNSpace::ReLU()          );
					if (args["--activator"]->array()[i]->string() == "TanH")            network.setActivator(new NNSpace::TanH()          );
				} else if (args["--activator"]->array()[i]->is_integer()) {
					if (args["--activator"]->array()[i]->integer() == NNSpace::ActivatorType::LINEAR)          network.setActivator(new NNSpace::Linear()        );
					if (args["--activator"]->array()[i]->integer() == NNSpace::ActivatorType::SIGMOID)         network.setActivator(new NNSpace::Sigmoid()       );
					if (args["--activator"]->array()[i]->integer() == NNSpace::ActivatorType::BIPOLAR_SIGMOID) network.setActivator(new NNSpace::BipolarSigmoid());
					if (args["--activator"]->array()[i]->integer() == NNSpace::ActivatorType::RELU)            network.setActivator(new NNSpace::ReLU()          );
					if (args["--activator"]->array()[i]->integer() == NNSpace::ActivatorType::TANH)            network.setActivator(new NNSpace::TanH()          );
				}
			}
		}
	}
	
	// Perform testing
	auto start_time = std::chrono::high_resolution_clock::now();
	unsigned long train_iterations = train_size;
	
	double rate = 0.5;
	std::vector<double> input(28 * 28);
	std::vector<double> output(10, 0);
	
	for (int i = train_offset; i < train_offset + train_size; ++i) {
		// Convert input
		for (int k = 0; k < 28 * 28; ++k)
			input[k] = (double) set.training_images[i][k] * (1.0 / 255.0);
		
		output[set.training_labels[i]] = 1.0;
		
		rate = NNSpace::backpropagation::train_error(network, Ltype, input, output, rate * rate_factor);
		
		output[set.training_labels[i]] = 0.0;
	}
	
	auto end_time = std::chrono::high_resolution_clock::now();
	
	// Do logging of the requested values
	if (args["--log"]) {
		if (args["--log"]->array_contains("TRAIN_TIME")) {
			
			// Calculate time used
			auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
			std::cout << "TRAIN_TIME=" << train_time << "ms" << std::endl;
		}
		if (args["--log"]->array_contains("TRAIN_ITERATIONS"))
			std::cout << "TRAIN_ITERATIONS=" << train_iterations << std::endl;
		if (args["--log"]->array_contains("TEST_MATCH")) 
			std::cout << "TEST_MATCH=" << NNSpace::Common::calculate_mnist_match(network, set, test_offset, test_size) << std::endl;
		if (args["--log"]->array_contains("TEST_ERROR_AVG")) 
			std::cout << "TEST_ERROR_AVG=" << NNSpace::Common::calculate_mnist_error(network, set, Ltype, test_offset, test_size) << std::endl;
		if (args["--log"]->array_contains("TEST_ERROR_MAX")) 
			std::cout << "TEST_ERROR_MAX=" << NNSpace::Common::calculate_mnist_error_max(network, set, Ltype, test_offset, test_size) << std::endl;
	}
	
	// Write network to file
	if (args["--output"] && args["--output"]->is_string())
		NNSpace::Common::write_network(network, args["--output"]->string());
	
	return 0;
};