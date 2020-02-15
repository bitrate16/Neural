#include <cstdlib>
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
 *  --steps=%        Amount of steps for training
 *  --output=%       Output file for the network
 *  --Ltype=%        L1 or L2
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
	dimensions.back() = 1;
	
	// Read weight info
	//  Input or 1.0
	double wD = args["--weight"] && args["--weight"]->is_real() ? args["--weight"]->real() : 1.0;
	
	// Read offsets flag
	bool offsets = args["--offsets"] && args["--offsets"]->get_boolean();
	
	// Read Ltype
	int Ltype = args["--Ltype"] ? args["--Ltype"]->get_integer() : 1;
	if (Ltype != 1 && Ltype != 2)
		Ltype = 1;
	
	// Read test & train set
	std::string train = args["--train"] && args["--train"]->is_string() ? args["--train"]->string() : "train.mset";
	std::string test  = args["--test"]  && args["--test"]->is_string()  ? args["--test"]->string()  : "test.mset";
	
	int steps = args["--steps"] ? args["--steps"]->get_integer() : 1;
	
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
	
	// Initialize positive step probability & step value
	std::vector<std::vector<std::vector<double>>> positive_probability(dimensions.size());
	std::vector<std::vector<std::vector<double>>> step(dimensions.size());
	for (int i = 0; i < dimensions.size() - 1; ++i) {
		positive_probability[i].resize(dimensions[i]);
		step[i].resize(dimensions[i]);
		for (int j = 0; j < dimensions[i + 1]; ++j) {
			positive_probability[i][j].resize(dimensions[i + 1], 0.5);
			step[i][j].resize(dimensions[i + 1], 0.5);
		}
	}
	
	// Error value before step
	double error_a = 0.5;
	// Errro value after step
	double error_b = 0.5;
	// Error change speed
	double error_d = 0.0;
	
	// For each step perform weight correction depending on selected direction
	for (int s = 0; s < steps; ++s) {
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
					if (probably_true(positive_probability[i][j]))
						network.W[d][i][j] += step[d][i][j];
					else
						network.W[d][i][j] -= step[d][i][j];
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
		if (args["--log"]->array_contains("TEST_ERROR"))
			std::cout << "TEST_ERROR=" << error_b << std::endl;
	}
	
	// Write network to file
	if (args["--output"] && args["--output"]->is_string())
		NNSpace::Common::write_network(networks[0], args["--output"]->string());
	
	return 0;
};