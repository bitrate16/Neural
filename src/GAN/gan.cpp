#include <iostream>
#include <vector>
#include <chrono>
#include <limits>

#include "NetTestCommon.h"

/*
 * (c) Copyright bitrate16 (GPLv3.0) 2020
 * 
 * Simple attempt of creating a GAN and teaching it to generate faces
 * Network consists of Generative and Testing networks.
 * Generative network takes input vector of 64 features and output the 3-channel image of size 100x100
 * Testing network takes the input of 3-channel image of size 100x100 and outputs vector of 64 features
 * Generative: { 64, 64*8, 100*100*3 }
 * Testing: { 100*100*3, 64*8,64 }
 * 
 * Arguments:
 *  --gen_layers=[%]     layer sizes for generator network
 *                       Not including the input, output layers. They are 64, %image_size^2 * 3.
 *  --gen_activator=[%]  Activator[i] function type for generator network
 *  --gen_weight=%       Weight dispersion for generator network
 *  --gen_output=%       Output file name for generator network
 *  --test_layers=[%]    layer sizes for test network
 *                       Not including the input, output layers. They are %image_size^2 * 3, 64.
 *  --test_activator=[%] Activator[i] function type for test network
 *  --test_weight=%      Weight dispersion for test network
 *  --test_output=%      Output file name for tester network
 *  --image_size=%       Size of the square of the scaled image
 *  --rate=%             Learning rate
 *  --train=%            Train set directory
 *  --train_size=%       Size of train set
 *  --train_offset=%     Offset of the train set
 *  --print              Enable debug print
 * 
 * 
 */

int main(int argc, const char** argv) {
	pargs::pargs args(argc, argv);
	
	// Read args
	double rate            = (args["--rate"] && args["--rate"]->is_real()) ? args["--rate"]->real() : 1.0;
	std::string train_path =  args["--train"] && args["--train"]->is_string() ? args["--train"]->string() : "";
	int train_size         = (args["--train_size"] && args["--train_size"]->is_integer()) ? args["--train_size"]->integer() : 100;
	int train_offset       = (args["--train_offset"] && args["--train_offset"]->is_integer()) ? args["--train_offset"]->integer() : 100;
	int image_size         = (args["--image_size"] && args["--image_size"]->is_integer()) ? args["--image_size"]->integer() : 50;
	bool print_flag        =  args["--print"];
	
	
	// G E N E R A T O R 
	
	std::string gen_output = args["--gen_output"] && args["--gen_output"]->is_string() ? args["--gen_output"]->string() : "generator.neetwook";
	
	// Input or 1.0
	double Wdg = args["--gen_weight"] && args["--gen_weight"]->is_real() ? args["--gen_weight"]->real() : 1.0;
	
	// Read input data for network definition
	std::vector<int> gen_dimensions = { 64, image_size * image_size * 3 };
	
	// Fill with layer dimensions
	if (args["--gen_layers"] && args["--gen_layers"]->is_array()) {
		gen_dimensions.resize(args["--gen_layers"]->array().size() + 2);
		
		for (int i = 0; i < args["--gen_layers"]->array().size(); ++i) {
			gen_dimensions[i + 1] = args["--gen_layers"]->array()[i]->integer();
			
			if (!gen_dimensions[i + 1])
				exit_message("Zero generator layer size");
		}
	}
	gen_dimensions.back() = image_size * image_size * 3;
	
	// Generate generator network
	NNSpace::MLNet generator;
	NNSpace::Common::generate_random_network(generator, dimensions, Wd, offsets);
	
	// Add activators (default is linear)
	if (args["--gen_activator"]) {
		if (args["--gen_activator"]->is_string()) 
			generator.setActivator(NNSpace::getActivatorByName(args["--gen_activator"]->string()));
		else if (args["--gen_activator"]->is_integer()) 
			generator.setActivator(NNSpace::getActivatorByType(args["--gen_activator"]->integer()));
		else if (args["--gen_activator"]->is_array()) {
			for (int i = 0; i < args["--gen_activator"]->array().size(); ++i) {
				if (args["--gen_activator"]->array()[i]->is_string()) 
					generator.setActivator(NNSpace::getActivatorByName(args["--gen_activator"]->array()[i]->string()));
				else if (args["--gen_activator"]->array()[i]->is_integer()) 
					generator.setActivator(NNSpace::getActivatorByType(args["--gen_activator"]->array[i]->integer()));
			}
		}
	}
	
	
	// T E S T 
	
	std::string test_output = args["--test_output"] && args["--test_output"]->is_string() ? args["--test_output"]->string() : "tester.neetwook";
	
	// Input or 1.0
	double Wdg = args["--test_weight"] && args["--test_weight"]->is_real() ? args["--test_weight"]->real() : 1.0;
	
	// Read input data for network definition
	std::vector<int> test_dimensions = { image_size * image_size * 3, 64 };
	
	// Fill with layer dimensions
	if (args["--test_layers"] && args["--test_layers"]->is_array()) {
		test_dimensions.resize(args["--test_layers"]->array().size() + 2);
		
		for (int i = 0; i < args["--test_layers"]->array().size(); ++i) {
			test_dimensions[i + 1] = args["--test_layers"]->array()[i]->integer();
			
			if (!test_dimensions[i + 1])
				exit_message("Zero tester layer size");
		}
	}
	test_dimensions.back() = 64;
	
	// Generate tester network
	NNSpace::MLNet tester;
	NNSpace::Common::generate_random_network(tester, dimensions, Wd, offsets);
	
	// Add activators (default is linear)
	if (args["--test_activator"]) {
		if (args["--test_activator"]->is_string()) 
			tester.setActivator(NNSpace::getActivatorByName(args["--test_activator"]->string()));
		else if (args["--test_activator"]->is_integer()) 
			tester.setActivator(NNSpace::getActivatorByType(args["--test_activator"]->integer()));
		else if (args["--test_activator"]->is_array()) {
			for (int i = 0; i < args["--test_activator"]->array().size(); ++i) {
				if (args["--test_activator"]->array()[i]->is_string()) 
					tester.setActivator(NNSpace::getActivatorByName(args["--test_activator"]->array()[i]->string()));
				else if (args["--test_activator"]->array()[i]->is_integer()) 
					tester.setActivator(NNSpace::getActivatorByType(args["--test_activator"]->array[i]->integer()));
			}
		}
	}
	
	
	// T R A I N
	
	// Iterate over directory
	int train_index = 0;
	
	std::vector<double> input_image(image_size * image_size * 2);
	
	for(auto& p: fs::directory_iterator(train_path)) {
		// Loop to the start example
		if (train_index < train_offset) {
			++train_index;
			continue;
		}
		
		// Limit amount
		if (train_index >= train_offset + train_size)
			break;
		
		
	}
	
	// Save networks
	NNSpace::Common::write_network(generator, gen_output);
	NNSpace::Common::write_network(tester, test_output);
	
	return 0;
}
	