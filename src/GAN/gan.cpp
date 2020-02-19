#include <iostream>
#include <vector>
#include <chrono>
#include <limits>

#include "train/backpropagation.h"
#include "NetTestCommon.h"
#include "pargs.h"

#include "CImg.h"

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
 *  --gen_output=%       Output file name for generator network
 *  --test_layers=[%]    layer sizes for test network
 *                       Not including the input, output layers. They are %image_size^2 * 3, 64.
 *  --test_output=%      Output file name for tester network
 *  --activator=%        Activator function type for all networks network
 *  --weight=%           Weight dispersion for all network
 *  --offsets=%          Enable offsets
 *  --image_size=%       Size of the square of the scaled image
 *  --rate=%             Learning rate
 *  --Ltype=%            L1 or L2
 *  --train=%            Train set directory
 *  --train_size=%       Size of train set
 *  --train_offset=%     Offset of the train set
 *  --print              Enable debug print
 * 
 * Make:
 * g++ src/GAN/gan.cpp -o bin/gan --std=c++17 -Iinclude -lstdc++fs -L/usr/X11R6/lib -lm -lpthread -lX11 
 * 
 * ./bin/gan --gen_layers=[200] --gen_output=networks/gan_gen.neetwook --test_layers=[200] --test_output=networks/gan_test.neetwook --activator=Sigmoid --weight=1.0 --offsets=true image_size=50 --rate=0.1 --train=data/UTKFace --train_size=100 --train_offset=0 --print
 * 
 */

// Simply prints out the message and exits.
inline void exit_message(const std::string& message) {
	if (message.size())
		std::cout << message << std::endl;
	exit(0);
};

int main(int argc, const char** argv) {
	pargs::pargs args(argc, argv);
	
	// Read args
	double rate            = (args["--rate"] && args["--rate"]->is_real()) ? args["--rate"]->real() : 1.0;
	std::string train_path =  args["--train"] && args["--train"]->is_string() ? args["--train"]->string() : "";
	int train_size         = (args["--train_size"] && args["--train_size"]->is_integer()) ? args["--train_size"]->integer() : 100;
	int train_offset       = (args["--train_offset"] && args["--train_offset"]->is_integer()) ? args["--train_offset"]->integer() : 100;
	int Ltype              = (args["--Ltype"] && args["--Ltype"]->is_integer()) ? args["--Ltype"]->integer() : 1;
	int image_size         = (args["--image_size"] && args["--image_size"]->is_integer()) ? args["--image_size"]->integer() : 50;
	bool offsets           =  args["--offsets"] && args["--offsets"]->get_boolean();
	bool print_flag        =  args["--print"];
	
	
	// S U P E R  N E T W O R K
	// G E N E R A T O R  &  T E S T E R
	
	std::string gen_output  = args["--gen_output"] && args["--gen_output"]->is_string() ? args["--gen_output"]->string() : "generator.neetwook";
	std::string test_output = args["--test_output"] && args["--test_output"]->is_string() ? args["--test_output"]->string() : "tester.neetwook";
	
	// Input or 1.0
	double Wd = args["--weight"] && args["--weight"]->is_real() ? args["--weight"]->real() : 1.0;
	
	// Read input data for network definition
	// Network as: Tester <-> Generator, stocked by feature output vector
	std::vector<int> dimensions = { image_size * image_size * 3 };
	// Count of tester layers, inclusing features layer and input image layer
	//  Used to slice networks
	int tester_layers_count = 1;
	
	// Fill with layer dimensions
	if (args["--test_layers"] && args["--test_layers"]->is_array()) {
		for (int i = 0; i < args["--test_layers"]->array().size(); ++i) {
			dimensions.push_back(args["--test_layers"]->array()[i]->integer());
			
			if (!dimensions.back())
				exit_message("Zero tester layer size");
			
			++tester_layers_count;
		}
	}
	
	dimensions.push_back(64);
	++tester_layers_count;
	
	if (args["--gen_layers"] && args["--gen_layers"]->is_array()) {
		for (int i = 0; i < args["--gen_layers"]->array().size(); ++i) {
			dimensions.push_back(args["--gen_layers"]->array()[i]->integer());
			
			if (!dimensions.back())
				exit_message("Zero tester layer size");
		}
	}
	
	dimensions.push_back(image_size * image_size * 3);
	
	if (print_flag) {
		std::cout << "Dimensions: [";
		for (int i = 0; i < dimensions.size(); ++i) {
			std::cout << dimensions[i];
			
			if (i != dimensions.size() - 1)
				std::cout << ", ";
		}
		std::cout << "]" << std::endl;
	}
	
	// Generate generator network
	NNSpace::MLNet network;
	NNSpace::Common::generate_random_network(network, dimensions, Wd, offsets);
	
	// Add activators (default is linear)
	if (args["--activator"])
		if (args["--activator"]->is_string()) 
			network.setActivator(NNSpace::getActivatorByName(args["--activator"]->string()));
		else if (args["--activator"]->is_integer()) 
			network.setActivator(NNSpace::getActivatorByType((NNSpace::ActivatorType) args["--activator"]->integer()));
	
	
	// T R A I N
	
	// Iterate over directory
	int train_index = 0;
	int train_iter  = 0;
	
	std::vector<double> input_image(image_size * image_size * 3);
	
	for(auto& p: std::experimental::filesystem::directory_iterator(train_path)) {
		// Loop to the start example
		if (train_index < train_offset) {
			++train_index;
			continue;
		}
		
		// Limit amount
		if (train_index >= train_offset + train_size)
			break;
		
		if (print_flag)
			std::cout << "Iteration: " << (train_iter++ + 1) << " / " << train_size << std::endl;
		
		// Read image
		cimg_library::CImg<unsigned char> image(p.path().c_str());
		
		// Resize to image_size
		image.resize(image_size, image_size);
		
		// Convert pixels
		for (int c = 0; c < 3; ++c)
			for (int i = 0; i < image_size; ++i)
				for (int j = 0; j < image_size; ++j)
					input_image[c * image_size * image_size + i * image_size + j] = ((double) image.atXY(i, j, c)) * (1.0 / 255.0);		
		
		// Train on thr same vector
		double self_error = NNSpace::backpropagation::train_error(network, Ltype, input_image, input_image, rate);
		
		if (print_flag)
			std::cout << "Self error: " << self_error << std::endl;
		
		++train_index;
	}
	
	// Save network as two networks
	{	
		if (print_flag)
			std::cout << "Writing generator: " << gen_output << std::endl;
	
		std::ofstream os;
		os.open(gen_output);
		
		if (os.fail()) 
			std::cout << "Failed write file " << gen_output << std::endl;
		else {
			os << dimensions.size() + 1 - tester_layers_count;
			os << std::endl;
			os << std::endl;
			
			for (int k = tester_layers_count - 1; k < dimensions.size(); ++k)
				os << dimensions[k] << ' ';
			os << std::endl;
			os << std::endl;
			
			for (int i = tester_layers_count - 1; i < dimensions.size() - 1; ++i)
				os << (int) network.activators[i]->getType() << ' ';
			os << std::endl;
			os << std::endl;
			
			for (int k = tester_layers_count - 1; k < dimensions.size() - 1; ++k) {
				for (int i = 0; i < dimensions[k]; ++i)
					for (int j = 0; j < dimensions[k + 1]; ++j) 
						os << network.W[k][i][j] << ' ';
				os << std::endl;
			}
			os << std::endl;
			
			for (int i = tester_layers_count - 1; i < dimensions.size() - 1; ++i) {
				for (int j = 0; j < dimensions[i + 1]; ++j)
					os << network.offsets[i][j] << ' ';
				os << std::endl;
			}
			os << std::endl;
			 
			os << network.enable_offsets;
			
			os.flush();
			os.close();
		}
	}
	
	{
		if (print_flag)
			std::cout << "Writing tester: " << test_output << std::endl;

		std::ofstream os;
		os.open(test_output);
		
		if (os.fail()) 
			std::cout << "Failed write file " << test_output << std::endl;
		else {
			os << tester_layers_count;
			os << std::endl;
			os << std::endl;
			
			for (int k = 0; k < tester_layers_count; ++k)
				os << dimensions[k] << ' ';
			os << std::endl;
			os << std::endl;
			
			for (int i = 0; i < tester_layers_count - 1; ++i)
				os << (int) network.activators[i]->getType() << ' ';
			os << std::endl;
			os << std::endl;
			
			for (int k = 0; k < tester_layers_count - 1; ++k) {
				for (int i = 0; i < dimensions[k]; ++i)
					for (int j = 0; j < dimensions[k + 1]; ++j) 
						os << network.W[k][i][j] << ' ';
				os << std::endl;
			}
			os << std::endl;
			
			for (int i = 0; i < tester_layers_count - 1; ++i) {
				for (int j = 0; j < dimensions[i + 1]; ++j)
					os << network.offsets[i][j] << ' ';
				os << std::endl;
			}
			os << std::endl;
			
			os << network.enable_offsets;
			
			os.flush();
			os.close();
		}
	}	
	
	return 0;
}
	