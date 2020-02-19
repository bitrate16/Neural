// pip install notebook

#include <iostream>
#include <vector>
#include <chrono>
#include <limits>
#include <cmath>

#include "train/backpropagation.h"
#include "NetTestCommon.h"
#include "pargs.h"

#include "CImg.h"

/*
 * (c) Copyright bitrate16 (GPLv3.0) 2020
 * 
 * Simple demo of GAN generation. Taker input network and generated output with N random features.
 * 
 * Arguments:
 *  --image_size=%       Size of the square of the scaled output image
 *  --network=%          Generative network location
 *  --output=%           Output folder for images
 *  --count=%            Amount of images to generate
 *  --print              Enable debug print
 * 
 * Make:
 * g++ src/GAN/gan_random.cpp -o bin/gan_random --std=c++17 -Iinclude -lstdc++fs -L/usr/X11R6/lib -lm -lpthread -lX11 
 * 
 * ./bin/gan_random --image_size=100 --network=networks/gan_gen.neetwook --output=gan_images --count=4 --print
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
	std::string path    =  args["--network"] && args["--network"]->is_string() ? args["--network"]->string() : "";
	std::string output  =  args["--output"] && args["--output"]->is_string() ? args["--output"]->string() : "";
	int count           = (args["--count"] && args["--count"]->is_integer()) ? args["--count"]->integer() : 1;
	int image_size      = (args["--image_size"] && args["--image_size"]->is_integer()) ? args["--image_size"]->integer() : 50;
	bool print_flag     =  args["--print"];
	
	// Generate generator network
	NNSpace::MLNet network;
	NNSpace::Common::read_network(network, path);
	
	
	// T R A I N
	
	int output_size = std::sqrt(network.dimensions.back() / 3);
	std::vector<double> input_features(64);
	std::vector<double> output_image(network.dimensions.back());
	
	for (int k = 0; k < count; ++k) {
		if (print_flag)
			std::cout << "Generating image #" << (k + 1) << std::endl;
		
		// Generate random features
		for (int j = 0; j < 64; ++j)
			input_features[j] = 0.5 + rand() * (0.5 / RAND_MAX);
		
		// Run network
		network.run(input_features, output_image);
		
		// Convert pixels
		cimg_library::CImg<unsigned char> img(output_size, output_size, 1, 3);
		
		for (int c = 0; c < 3; ++c)
			for (int i = 0; i < output_size; ++i)
				for (int j = 0; j < output_size; ++j)
					img.atXY(i, j, c) = output_image[c * output_size * output_size + i * output_size + j] * 255.0;
		
		//img.display("ur mom gay");
		img.resize(image_size, image_size);
		
		std::string filename = output + "/gan_img_ " + std::to_string(k) + ".jpg";
		
		img.save(filename.c_str());
	}
	
	return 0;
}
	