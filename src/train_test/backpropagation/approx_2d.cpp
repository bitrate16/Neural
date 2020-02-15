#include <vector>
#include <chrono>

#include "train/backpropagation.h"
#include "NetTestCommon.h"
#include "pargs.h"

/*
 * Performs simpla testing of passed amount of networks with passed aguments.
 * Arguments:
 *  --layers=[%]     layer sizes
 *                   Not including the input, output layers. They are 1, 1.
 *  --activator=[%]  Activator[i] function type
 *  --weight=%       Weight dispersion
 *  --offsets=%      Enable offfsets flag
 *  --train=%        Input train set
 *  --test=%         Input test set
 *  --output=%       Output file for the network
 *  --error=%        L1 or L2
 *  --log=[%]        Log type (TRAIN_TIME, TRAIN_OPERATIONS, TRAIN_ITERATIONS, TEST_ERROR)
 *
 * Make:
 * g++ src/train_test/backpropagation/approx_2d.cpp -o bin/backpropagation_approx_2d -O3 --std=c++11 -Iinclude -lstdc++fs
 *
 * Example:
 * ./bin/backpropagation_approx_2d --layers=[3] --activator=TanH --weight=1.0 --train=data/sin_1000.mset --test=data/sin_100.mset --output=networks/approx_sin.neetwook --log=[TRAIN_TIME,TEST_ERROR,TRAIN_ITERATIONS]
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
	std::vector<int> dimensions = { 1, 1 };
	
	// Fill with layer dimensions
	if (args["--layers"] && args["--layers"]->is_array()) {
		dimensions.resize(args["--layers"]->array().size() + 2);
		
		for (int i = 0; i < args["--layers"]->array().size(); ++i) {
			dimensions[i] = args["--layers"]->array()[i]->integer();
			
			if (!dimensions[i])
				exit_message("Zero layer size");
		}
	}
	dimensions.back() = 1;
	
	// Read weight info
	//  Input or 1.0
	double wD = args["--weight"] && args["--weight"]->is_real() ? args["--weight"]->real() : 1.0;
	
	// Read offsets flag
	bool offsets = args["--offsets"] && args["--offsets"]->get_boolean();
	
	// Read Ltype
	int Ltype = args["--error"] ? args["--error"]->get_integer() : 1;
	if (Ltype != 1 && Ltype != 2)
		Ltype = 1;
	
	// Read test & train set
	std::string train = args["--train"] && args["--train"]->is_string() ? args["--train"]->string() : "train.mset";
	std::string test  = args["--test"]  && args["--test"]->is_string()  ? args["--test"]->string()  : "test.mset";
	
	// Read train set data
	std::vector<std::pair<double, double>> train_set;
	if (!NNSpace::Common::read_approx_set(train_set, train))
		exit_message("Set " + train + " not found");
	
	std::vector<std::pair<double, double>> test_set;
	if (!NNSpace::Common::read_approx_set(test_set,  test))
		exit_message("Set " + test + " not found");
	
	// Generate network
	NNSpace::MLNet network;
	NNSpace::Common::generate_random_network(network, dimensions, wD, offsets);
	
	// Add activators (default is linear)
	if (args["--activator"]) {
		if (args["--activator"]->is_string()) {
			if (args["--activator"]->string() == "LINEAR")          network.setActivator(new NNSpace::Linear()        );
			if (args["--activator"]->string() == "SIGMOID")         network.setActivator(new NNSpace::Sigmoid()       );
			if (args["--activator"]->string() == "BIPOLAR_SIGMOID") network.setActivator(new NNSpace::BipolarSigmoid());
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
					if (args["--activator"]->array()[i]->string() == "LINEAR")          network.setActivator(new NNSpace::Linear()        );
					if (args["--activator"]->array()[i]->string() == "SIGMOID")         network.setActivator(new NNSpace::Sigmoid()       );
					if (args["--activator"]->array()[i]->string() == "BIPOLAR_SIGMOID") network.setActivator(new NNSpace::BipolarSigmoid());
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
	unsigned long train_iterations = train_set.size();
	
	double rate = 0.5;
	std::vector<double> input(1);
	std::vector<double> output(1);
	
	for (auto& p : train_set) {
		input[0]  = p.first;
		output[0] = p.second;
		rate = NNSpace::backpropagation::train_error(network, Ltype, input, output, rate);
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
		if (args["--log"]->array_contains("TEST_ERROR")) {
	
			// Calculate error value
			double error = NNSpace::Common::calculate_approx_error(network, test_set, Ltype);
			std::cout << "TEST_ERROR=" << error << std::endl;
		}
	}
	
	// Write network to file
	if (args["--output"] && args["--output"]->is_string())
		NNSpace::Common::write_network(network, args["--output"]->string());
	
	return 0;
};