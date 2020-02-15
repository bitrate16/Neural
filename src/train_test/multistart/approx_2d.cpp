#include <iostream>
#include <vector>
#include <chrono>
#include <limits>

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
 *  --Ltype=%        L1 or L2
 *  --networks=%     Cmount of startup networks
 *  --log=[%]        Log type (TRAIN_TIME, TRAIN_OPERATIONS, TRAIN_ITERATIONS, TEST_ERROR)
 *
 * Make:
 * g++ src/train_test/multistart/approx_2d.cpp -o bin/multistart_approx_2d -O3 --std=c++11 -Iinclude -lstdc++fs
 *
 * Example:
 * ./bin/multistart_approx_2d --networks=16 --layers=[3] --activator=TanH --weight=1.0 --train=data/sin_1000.mset --test=data/sin_100.mset --output=networks/approx_sin.neetwook --log=[TRAIN_TIME,TEST_ERROR,TRAIN_ITERATIONS]
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
			dimensions[i + 1] = args["--layers"]->array()[i]->integer();
			
			if (!dimensions[i + 1])
				exit_message("Zero layer size");
		}
	}
	dimensions.back() = 1;
	
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
	int Ltype = args["--Ltype"] ? args["--Ltype"]->get_integer() : 1;
	if (Ltype != 1 && Ltype != 2)
		Ltype = 1;
	
	// Read test & train set
	std::string train = args["--train"] && args["--train"]->is_string() ? args["--train"]->string() : "train.mset";
	std::string test  = args["--test"]  && args["--test"]->is_string()  ? args["--test"]->string()  : "test.mset";
	
	// Calculate Af to split networks
	int Af = 0;
	{
		unsigned int size = count;
		while (size >>= 1) 
			++Af;
	}
	
	// Read train set data
	std::vector<std::vector<std::pair<double, double>>> train_sets;
	{
		std::vector<std::pair<double, double>> train_set;
		if (!NNSpace::Common::read_approx_set(train_set, train))
			exit_message("Set " + train + " not found");
		
		if (train_set.size() / Af == 0)
			exit_message("Not enough train set size");
		
		NNSpace::Common::split_approx_set(train_sets, train_set, train_set.size() / Af);
	}
	
	std::vector<std::pair<double, double>> test_set;
	if (!NNSpace::Common::read_approx_set(test_set,  test))
		exit_message("Set " + test + " not found");
	
	// Generate network
	std::vector<NNSpace::MLNet> networks;
	NNSpace::Common::generate_random_networks(networks, dimensions, wD, offsets, count);	
	
	// Add activators (default is linear)
	if (args["--activator"]) {
		for (int i = 0; i < networks.size(); ++i) {
			if (args["--activator"]->is_string()) {
				if (args["--activator"]->string() == "LINEAR")          networks[i].setActivator(new NNSpace::Linear()        );
				if (args["--activator"]->string() == "SIGMOID")         networks[i].setActivator(new NNSpace::Sigmoid()       );
				if (args["--activator"]->string() == "BIPOLAR_SIGMOID") networks[i].setActivator(new NNSpace::BipolarSigmoid());
				if (args["--activator"]->string() == "ReLU")            networks[i].setActivator(new NNSpace::ReLU()          );
				if (args["--activator"]->string() == "TanH")            networks[i].setActivator(new NNSpace::TanH()          );
			} else if (args["--activator"]->is_integer()) {
				if (args["--activator"]->integer() == NNSpace::ActivatorType::LINEAR)          networks[i].setActivator(new NNSpace::Linear()        );
				if (args["--activator"]->integer() == NNSpace::ActivatorType::SIGMOID)         networks[i].setActivator(new NNSpace::Sigmoid()       );
				if (args["--activator"]->integer() == NNSpace::ActivatorType::BIPOLAR_SIGMOID) networks[i].setActivator(new NNSpace::BipolarSigmoid());
				if (args["--activator"]->integer() == NNSpace::ActivatorType::RELU)            networks[i].setActivator(new NNSpace::ReLU()          );
				if (args["--activator"]->integer() == NNSpace::ActivatorType::TANH)            networks[i].setActivator(new NNSpace::TanH()          );
			} else if (args["--activator"]->is_array()) {
				for (int i = 0; i < args["--activator"]->array().size(); ++i) {
					if (args["--activator"]->array()[i]->is_string()) {
						if (args["--activator"]->array()[i]->string() == "LINEAR")          networks[i].setActivator(new NNSpace::Linear()        );
						if (args["--activator"]->array()[i]->string() == "SIGMOID")         networks[i].setActivator(new NNSpace::Sigmoid()       );
						if (args["--activator"]->array()[i]->string() == "BIPOLAR_SIGMOID") networks[i].setActivator(new NNSpace::BipolarSigmoid());
						if (args["--activator"]->array()[i]->string() == "ReLU")            networks[i].setActivator(new NNSpace::ReLU()          );
						if (args["--activator"]->array()[i]->string() == "TanH")            networks[i].setActivator(new NNSpace::TanH()          );
					} else if (args["--activator"]->array()[i]->is_integer()) {
						if (args["--activator"]->array()[i]->integer() == NNSpace::ActivatorType::LINEAR)          networks[i].setActivator(new NNSpace::Linear()        );
						if (args["--activator"]->array()[i]->integer() == NNSpace::ActivatorType::SIGMOID)         networks[i].setActivator(new NNSpace::Sigmoid()       );
						if (args["--activator"]->array()[i]->integer() == NNSpace::ActivatorType::BIPOLAR_SIGMOID) networks[i].setActivator(new NNSpace::BipolarSigmoid());
						if (args["--activator"]->array()[i]->integer() == NNSpace::ActivatorType::RELU)            networks[i].setActivator(new NNSpace::ReLU()          );
						if (args["--activator"]->array()[i]->integer() == NNSpace::ActivatorType::TANH)            networks[i].setActivator(new NNSpace::TanH()          );
					}
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
	// a - before train
	// b - after train
	// d - error delta
	std::vector<double> errors_a(networks.size(), 0.5);
	std::vector<double> errors_b(networks.size(), 0.5);
	std::vector<double> errors_d(networks.size(), 0.5);
	// Maximal error value 
	double error_max = 0.0;
	// Minimal Error delta
	double varie_min = 2.0;
	// Index array for sorting the networks by their errro value
	std::vector<int> index_array(networks.size());
	
	// Iterate over epochs
	for (int epo = 0; epo < Af; ++epo) {
		error_max      = 0.0;
		varie_min      = std::numeric_limits<long double>::max();
		
		for (int k = 0; k < networks.size(); ++k) {
			errors_a[k]    = errors_b[k];
			index_array[k] = k;
			
			// Train with backpropagation
			for (auto& p : train_sets[epo]) {
				input[0]  = p.first;
				output[0] = p.second;
				rates[k]  = NNSpace::backpropagation::train_error(networks[k], Ltype, input, output, rates[k]);
			}
			
			train_iterations += train_sets[epo].size();
			
			// Calculate error value on testing set
			errors_b[k] = NNSpace::Common::calculate_approx_error(networks[k], test_set, Ltype);
			errors_d[k] = errors_b[k] - errors_a[k];
			
			// Update min/max
			if (varie_min > errors_d[k])
				varie_min = errors_d[k];
			if (error_max < errors_b[k])
				error_max = errors_b[k];
		}
		
		// Order networks by their testing error value
		std::sort(index_array.begin(), index_array.end(), [&errors_d, &errors_b, &error_max, &varie_min](const int& a, const int& b) {
			return 	(error_max - errors_d[a] + errors_b[a] - varie_min)  // Distance from A to error values
					>
					(error_max - errors_d[b] + errors_b[b] - varie_min); // Distance from B to error values
		});
		
		// Reduce amount of networks by 2
		int slice_size = index_array.size() / 2 + index_array.size() % 2;
		
		// Slice half of an array and sort because removing 
		//  unordered indexes is undefined behaviour.
		std::vector<int> index_array_ordered(index_array.begin(), index_array.begin() + slice_size);
		std::sort(index_array_ordered.begin(), index_array_ordered.end(), [](const int& a, const int& b) { return a > b; });
		
		for (int i = 0; i < slice_size; ++i) {
			networks.erase(networks.begin() + index_array_ordered[i]);
			rates   .erase(rates   .begin() + index_array_ordered[i]);
			errors_a.erase(errors_a.begin() + index_array_ordered[i]);
			errors_b.erase(errors_b.begin() + index_array_ordered[i]);
			errors_d.erase(errors_d.begin() + index_array_ordered[i]);
		}
		
		index_array.resize(index_array.size() - slice_size);
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
		if (args["--log"]->array_contains("TEST_ERROR"))
			std::cout << "TEST_ERROR=" << errors_b[0] << std::endl;
	}
	
	// Write network to file
	if (args["--output"] && args["--output"]->is_string())
		NNSpace::Common::write_network(networks[0], args["--output"]->string());
	
	return 0;
};