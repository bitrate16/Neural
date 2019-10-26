#include "Network.h"
#include "MultiLayerNetwork.h"

#include "mnist/mnist_reader.hpp"

#include <iostream>
#include <iomanip>

// Tweaks
#define MNIST_DATA_LOCATION "input"
#define NETWORK_SERIALIZE_PATH "output/digits_ml.neetwook"
#define TRAIN_SET_SIZE dataset.training_images.size()
#define TRAIN_RATE 0.03
#define DEEP_LAYER_SET 19 * 19, 10 * 10
// #define DEEP_LAYER_SET 25 * 25, 20 * 20, 15 * 15, 10 * 10, 5 * 5
#define NETWORK_ACTIVATOR_FUNCTION Sigmoid

// #define TRAIN_AND_RUN
 #define TRAIN
// #define TRAIN_ERROR
 #define RUN

// Perform train of the network & running on tests
void train_and_run_main() {
	std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

	mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;
	
	NNSpace::MLNetwork network({ 28 * 28, DEEP_LAYER_SET, 10 });
	network.randomize();
	network.setFunction(new NNSpace::NETWORK_ACTIVATOR_FUNCTION());
	
	for (int i = 0; i < TRAIN_SET_SIZE; ++i) {
		std::cout << "Train " << i << " / " << TRAIN_SET_SIZE << std::endl;
		std::vector<double> input(28 * 28);
		std::vector<double> output(10);
		
		for (int j = 0; j < 28 * 28; ++j)
			input[j] = dataset.training_images[i][j] / 255.0;
		
		output[dataset.training_labels[i]] = 1.0;
		
		network.train(input, output, TRAIN_RATE);
	}
	
	int passed_amount = 0;
	for (int i = 0; i < dataset.test_images.size(); ++i) {
		std::vector<double> input(28 * 28);
		
		for (int j = 0; j < 28 * 28; ++j)
			input[j] = dataset.test_images[i][j] / 255.0;
		
		std::vector<double> output = network.run(input);
		
		std::cout << "EXPECT: " << (int) dataset.test_labels[i] << ", GOT: ";
		
		int max = 0;
		double maxv = -2.0;
		for (int i = 0; i < output.size(); ++i) {
			if (maxv < output[i]) {
				maxv = output[i];
				max = i;
			}
		}
		
		std::cout << max << ", ";
		if (max != dataset.test_labels[i])
			std::cout << "MISS " << maxv;
		else {
			++passed_amount;
			std::cout << "PASS";
		}
		
		std::cout << std::endl;
	}
	
	std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
	std::cout.precision(2);
	std::cout << "RESULT: " << passed_amount << '/' << dataset.test_images.size() << " [" << (100.0 * (double) passed_amount / (double)dataset.test_images.size()) << "%]" << std::endl;
};

// Perform train of the network & serialize
void train_and_serialize() {
	std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

	mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;
	
	NNSpace::MLNetwork network({ 28 * 28, DEEP_LAYER_SET, 10 });
	network.randomize();
	network.setFunction(new NNSpace::NETWORK_ACTIVATOR_FUNCTION());
	
	for (int i = 0; i < TRAIN_SET_SIZE; ++i) {
		std::cout << "Train " << i << " / " << TRAIN_SET_SIZE << std::endl;
		std::vector<double> input(28 * 28);
		std::vector<double> output(10);
		
		for (int j = 0; j < 28 * 28; ++j)
			input[j] = dataset.training_images[i][j] / 255.0;
		
		output[dataset.training_labels[i]] = 1.0;
		
		network.train(input, output, TRAIN_RATE);
	}
	
	std::cout << "Serializing network to " << NETWORK_SERIALIZE_PATH << std::endl;
	std::ofstream of;
	of.open(NETWORK_SERIALIZE_PATH);
	if (of.fail()) {
		std::cout << "File " << NETWORK_SERIALIZE_PATH << " open failed" << std::endl;
		return;
	}
	network.serialize(of);
	of.flush();
	of.close();
};

// Perform train of the network & serialize & print error values during train
void train_error_and_serialize() {
	std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

	mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;
	
	NNSpace::MLNetwork network({ 28 * 28, DEEP_LAYER_SET, 10 });
	network.randomize();
	network.setFunction(new NNSpace::NETWORK_ACTIVATOR_FUNCTION());
	
	std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
	std::cout.precision(4);
	
	for (int i = 0; i < TRAIN_SET_SIZE; ++i) {
		std::cout << "Train " << i << " / " << TRAIN_SET_SIZE << ' ';
		std::vector<double> input(28 * 28);
		std::vector<double> output(10);
		
		for (int j = 0; j < 28 * 28; ++j)
			input[j] = dataset.training_images[i][j] / 255.0;
		
		output[dataset.training_labels[i]] = 1.0;
		
		std::cout << network.train_error(input, output, TRAIN_RATE) << std::endl;;
	}
	
	std::cout << "Serializing network to " << NETWORK_SERIALIZE_PATH << std::endl;
	std::ofstream of;
	of.open(NETWORK_SERIALIZE_PATH);
	if (of.fail()) {
		std::cout << "File " << NETWORK_SERIALIZE_PATH << " open failed" << std::endl;
		return;
	}
	network.serialize(of);
	of.flush();
	of.close();
};

// Deserialize data and run
void deserialize_and_run() {
	std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

	mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;
	
	NNSpace::MLNetwork network({ 28 * 28, DEEP_LAYER_SET, 10 });
	network.setFunction(new NNSpace::NETWORK_ACTIVATOR_FUNCTION());
	
	std::cout << "Deserializing network from " << NETWORK_SERIALIZE_PATH << std::endl;
	std::ifstream ifs;
	ifs.open(NETWORK_SERIALIZE_PATH);
	if (ifs.fail()) {
		std::cout << "File " << NETWORK_SERIALIZE_PATH << " not found" << std::endl;
		return;
	}
	if (!network.deserialize(ifs)) {
		std::cout << "Deserialize failed" << std::endl;
		ifs.close();
		return;
	}
	ifs.close();
	
	int passed_amount = 0;
	for (int i = 0; i < dataset.test_images.size(); ++i) {
		std::vector<double> input(28 * 28);
		
		for (int j = 0; j < 28 * 28; ++j)
			input[j] = dataset.test_images[i][j] / 255.0;
		
		std::vector<double> output = network.run(input);
		
		std::cout << "EXPECT: " << (int) dataset.test_labels[i] << ", GOT: ";
		
		int max = 0;
		double maxv = -2.0;
		for (int i = 0; i < output.size(); ++i) {
			if (maxv < output[i]) {
				maxv = output[i];
				max = i;
			}
		}
		
		std::cout << max << ", ";
		if (max != dataset.test_labels[i])
			std::cout << "MISS " << maxv;
		else {
			++passed_amount;
			std::cout << "PASS";
		}
		
		std::cout << std::endl;
	}
	
	std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
	std::cout.precision(2);
	std::cout << "RESULT: " << passed_amount << '/' << dataset.test_images.size() << " [" << (100.0 * (double) passed_amount / (double)dataset.test_images.size()) << "%]" << std::endl;
};

// bash c.sh "-O3" src/main_layered

int main(int argc, char** argv) {

#ifdef TRAIN_AND_RUN
	train_and_run_main();
#else
#ifdef TRAIN
	train_and_serialize();
#endif
#ifdef TRAIN_ERROR
	train_error_and_serialize();
#endif
#ifdef RUN
	deserialize_and_run();
#endif
#endif

	return 0;
};