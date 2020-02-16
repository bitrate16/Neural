#include <iostream>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <limits>

#include "NetTestCommon.h"
#include "pargs.h"

/*
 * (c) Copyright bitrate16 (GPLv3.0) 2020
 * Performs simpla testing of passed amount of networks with passed aguments.
 * Testing on MNIST Digit recognition
 * Arguments:
 *  --layers=[%]     layer sizes
 *                   Not including the input, output layers. They are 1, 1.
 *  --activator=[%]  Activator[i] function type
 *  --weight=%       Weight dispersion
 *  --offsets=%      Enable offfsets flag
 *  --train=%        Input train set
 *  --test=%         Input test set
 *  --train_size=%   Amount of digits taken from train set
 *  --test_size=%    Amount of digits taken from test set
 *  --train_offset=% Offset value for train set
 *  --test_offset=%  Offset value for test set
 *  --steps=%        Amount of steps for training
 *  --output=%       Output file for the network
 *  --Ltype=%        L1 or L2
 *  --log=[%]        Log type (TRAIN_TIME, TRAIN_OPERATIONS, TRAIN_ITERATIONS, TEST_ERROR_AVG, TEST_ERROR_MAX)
 *
 * Make:
 * g++ src/train_test/random_search/mnist.cpp -o bin/random_search_mnist -O3 --std=c++17 -Iinclude -lstdc++fs
 *
 * Example:
 * ./bin/random_search_mnist --steps=16 --layers=[3] --offsets=true --activator=Sigmoid --weight=1.0 --train_size=10000 --test_size=100 --mnist=data/mnist --output=networks/mnist_test.neetwook --log=[TRAIN_TIME,TEST_ERROR_AVG,TEST_ERROR_MAX,TRAIN_ITERATIONS,TEST_MATCH]
 *
 * ./bin/random_search_mnist --steps=16 --layers=[5] --offsets=true --activator=TanH --weight=10.0 --train=data/sin_1000.mset --test=data/sin_100.mset --output=networks/mnist_test.neetwook --log=[TRAIN_TIME,TEST_ERROR_AVG,TEST_ERROR_MAX,TRAIN_ITERATIONS,TEST_MATCH]
 */

// Simply prints out the message and exits.
inline void exit_message(const std::string& message) {
	if (message.size())
		std::cout << message << std::endl;
	exit(0);
};

// Generate boolean with given probability
bool probably_true(double p) {
    return rand() * (1.0 / (RAND_MAX + 1.0)) < p;
};

int main(int argc, const char** argv) {
	pargs::pargs args(argc, argv);
	
	// Read input data for network definition
	std::vector<int> dimensions = { 1, 1 };
	
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
	
	int steps = args["--steps"] ? args["--steps"]->get_integer() : 1;
	
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
	unsigned long train_iterations = 0;
	
	// Initialize positive step probability & step value
	std::vector<std::vector<std::vector<double>>> positive_probability(dimensions.size() - 1);
	std::vector<std::vector<std::vector<double>>> step(dimensions.size() - 1);
	for (int i = 0; i < dimensions.size() - 1; ++i) {
		positive_probability[i].resize(dimensions[i]);
		step[i].resize(dimensions[i]);
		for (int j = 0; j < dimensions[i]; ++j) {
			positive_probability[i][j].resize(dimensions[i + 1], 0.5);
			step[i][j].resize(dimensions[i + 1], wD / 2.0);
		}
	}
	
	std::vector<std::vector<double>> positive_offset_probability(dimensions.size() - 1);
	std::vector<std::vector<double>> offset_step(dimensions.size() - 1);
	if (offsets)
		for (int i = 0; i < dimensions.size() - 1; ++i) {
			positive_offset_probability[i].resize(dimensions[i + 1], 0.5);
			offset_step[i].resize(dimensions[i + 1], 0.5);
		}
	
	// Error value before step
	double error_a = 0.5;
	// Errro value after step
	double error_b = 0.5;
	// Error change speed
	double error_d = 0.0;
	
	// For each step perform weight correction depending on selected direction
	for (int s = 0; s < steps; ++s) {
		++train_iterations;
		error_a = error_b;
		
		// Perform correction depending on probability
		for (int d = 0; d < dimensions.size() - 1; ++d)
			for (int i = 0; i < dimensions[d]; ++i)
				for (int j = 0; j < dimensions[d + 1]; ++j) {
					
					// Change probability depending on error_d and error_b after previous step
					if (s) {
						// De - Delta error
						// D  - step
						// e  - last error
						// 
						// Probability calibration:
						// 	Pi+1 = Pi * (1 + De)
						// 
						// Step calibration (two variants) (De > 0):
						//  I:  Di+1 = Di * 2 * (1 - De) * 2 * e
						//  II: Di+1 = 2 * (1 - e)
						
						#define METHOD_I
						// #define METHOD_II
						
						#ifdef METHOD_I
							if (error_d > 0)
								step[d][i][j] = step[d][i][j] * 2.0 * (1.0 - error_d) * 2.0 * error_b;
						#endif
						#ifdef METHOD_II
							if (error_d > 0)
								step[d][i][j] = step[d][i][j] * 2.0 * error_b;
						#endif
						
						positive_probability[d][i][j] = positive_probability[d][i][j] * (1.0 + error_d);
						if (positive_probability[d][i][j] > 1.0)
							positive_probability[d][i][j] = 1.0;
					}
					
					// Generate random direction & step on it
					if (probably_true(positive_probability[d][i][j]))
						network.W[d][i][j] += step[d][i][j];
					else
						network.W[d][i][j] -= step[d][i][j];
				}
			
		// Perform offset correction depending on probability
		if (offsets)
			for (int i = 0; i < dimensions.size() - 1; ++i)
				for (int j = 0; j < dimensions[i + 1]; ++j) {
					// Change probability depending on error_d and error_b after previous step
					if (s) {						
						#ifdef METHOD_I
							if (error_d > 0)
								offset_step[i][j] = offset_step[i][j] * 2.0 * (1.0 - error_d) * 2.0 * error_b;
						#endif
						#ifdef METHOD_II
							if (error_d > 0)
								offset_step[i][j] = offset_step[i][j] * 2.0 * error_b;
						#endif
						
						positive_offset_probability[i][j] = positive_offset_probability[i][j] * (1.0 + error_d);
						if (positive_offset_probability[i][j] > 1.0)
							positive_offset_probability[i][j] = 1.0;
					}
					
					// Generate random direction & step on it
					if (probably_true(positive_offset_probability[i][j]))
						network.offsets[i][j] += offset_step[i][j];
					else
						network.offsets[i][j] -= offset_step[i][j];
				}
		
		// Calculate error value after step (teach_set)
		error_b = NNSpace::Common::calculate_approx_error(network, train_set, Ltype);
		
		// Calculate error change speed
		error_d = error_b - error_a;
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
			std::cout << "TEST_ERROR_AVG=" << NNSpace::Common::calculate_mnist_match(network, set, test_offset, test_size) << std::endl;
		if (args["--log"]->array_contains("TEST_ERROR_AVG"))
			std::cout << "TEST_ERROR=" << error_b << std::endl;
		if (args["--log"]->array_contains("TEST_ERROR_MAX")) 
			std::cout << "TEST_ERROR_MAX=" << NNSpace::Common::calculate_mnist_error_max(network, set, Ltype, test_offset, test_size) << std::endl;
	}
	
	// Write network to file
	if (args["--output"] && args["--output"]->is_string())
		NNSpace::Common::write_network(network, args["--output"]->string());
	
	return 0;
};