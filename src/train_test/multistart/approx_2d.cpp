#include <vector>
#include <chrono>

#include "train/backpropagation.h"
#include "NetTestCommon.h"
#include "pargs.h"

/*
 * Performs simpla testing of passed amount of networks with passed aguments.
 * For passed networks count, calculates Af as log2(amount).
 *  Passed set is being split into Af subsets (S[i]) and training being performed:
 *   For each epoch [i] networks are trained on S[i].
 *   For each epoch [i] calculating error value for each network.
 *   For each epoch [i] networks count is reduced by 2, removing networks with the largest error value.
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
 *  --networks=%     Cmount of startup networks
 *  --log=[%]        Log type (TRAIN_TIME, TRAIN_OPERATIONS, TRAIN_ITERATIONS, TEST_ERROR)
 *
 * Make:
 * g++ src/train_test/backpropagation/approx_2d.cpp -o bin/backpropagation_approx_2d -O3 --std=c++11 -Iinclude -lstdc++fs
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
	
	// Read weight info
	//  Input or 1.0
	double wD = args["--weight"] && args["--weight"]->is_real() ? args["--weight"]->real() : 1.0;
	
	// Read offsets flag
	bool offsets = args["--offsets"] && args["--offsets"]->get_boolean();
	
	// Read networks amount
	int count = args["--networks"] && args["--networks"]->is_integer() ? args["--networks"]->integer() : 1;
	if (count <= 0)
		exit_message("Invalid networks count");
	
	// Read Ltype
	int Ltype = args["--error"] ? args["--error"]->get_integer() : 1;
	if (Ltype != 1 && Ltype != 2)
		Ltype = 1;
	
	// Read test & train set
	std::string train = args["--train"] && args["--train"]->is_string() ? args["--train"]->string() : "train.mset";
	std::string test  = args["--test"]  && args["--test"]->is_string()  ? args["--test"]->string()  : "test.mset";
	
	// Calculate Af to split networks
	int Af = 0;
	{
		unsigned int size = count;
		while (size) {
			++Af;
			size >>= 1;
		}
	}
	
	// Read train set data
	std::vector<std::vector<std::pair<double, double>>> train_sets;
	{
		std::vector<std::pair<double, double>> train_set;
		if (!NNSpace::Common::read_approx_set(train_set, train))
			exit_message("Set " + train + " not found");
		
		if (rain_set.size() / Af == 0)
			exit_message("Not enough train set size");
		
		NNSpace::Common::split_approx_set(train_sets, train_set, train_set.size() / Af);
	}
	
	std::vector<std::pair<double, double>> test_set;
	if (!NNSpace::Common::read_approx_set(test_set,  test))
		exit_message("Set " + test + " not found");
	
	// Generate network
	std::vector<NNSpace::MLNet> network;
	NNSpace::Common::generate_random_networks(network, dimensions, wD, offsets, count);
	
	
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
	unsigned long train_iterations = 0;
	
	std::vector<double> input(1);
	std::vector<double> output(1);
	
	// Training rate value
	std::vector<double> rates(networks.size(), 0.5);
	// Testing error value
	std::vector<double> errors(networks.size(), 0.5);
	
	// Iterate over epochs
	for (int epo = 0; epo < Af; ++epo) {
		for (int k = 0; k < networks.size(); ++k) {
			
			// Train with backpropagation
			for (auto& p : train_sets[epo]) {
				input[0]  = p.first;
				output[0] = p.second;
				rates[k]  = NNSpace::backpropagation::train_error(networks[k], Ltype, input, output, rates[k]);
			}
			
			train_iterations += train_sets[epo].size();
			
			// Calculate error value on testing set
			errors[k] = NNSpace::Common::calculate_approx_error(networks[k], test_set, Ltype);
		}
		
		// Order networks by their testing error value
		
	}
	
	auto end_time = std::chrono::high_resolution_clock::now();
	
	// Do logging of the requested values
	if (args["--log"]) {
		if (args["--log"]->array_contains("TRAIN_TIME")) {
			
			// Calculate time used
			auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(start_time - end_time).count();
			std::cout << "TRAIN_TIME=" << train_time << std::endl;
		}
		if (args["--log"]->array_contains("TRAIN_ITERATIONS"))
			std::cout << "TRAIN_ITERATIONS=" << train_iterations << std::endl;
		if (args["--log"]->array_contains("TEST_ERROR")) {
			std::cout << "TEST_ERROR=" << errors[0] << std::endl;
	}
	
	// Write network to file
	if (args["--output"] && args["--output"]->is_string())
		NNSpace::Common::write_network(networks[0], args["--output"]->string());
	
	return 0;
};