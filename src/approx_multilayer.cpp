#include "Network.h"
#include "MultiLayerNetwork.h"

#include <iostream>
#include <random>
#include <fstream>
#include <iomanip>
#include <algorithm>

// Tweaks
#define MNIST_DATA_LOCATION "input"
#define NETWORK_SERIALIZE_PATH "output/approx.neetwook"

// Training & testing properties
#define START_TRAIN_RATE  0.01
#define SCALE_TRAIN_RATE  0.5
#define TRAIN_SET_SIZE    10000
#define TEST_SET_SIZE     100
#define ACCURATE_SET_SIZE 100
#define TRAIN_ALGO_ID     0
#define INIT_ALGO_ID      0

#define FUNCTION_START    0.0
#define FUNCTION_END      1.0
#define FUNCTION_FUNC     std::sin(2.0 * 3.141528 * t) * 0.5 + 0.5

#define NETWORK_ACTIVATOR_FUNCTION TanH
#define NETWORK_TOPO               { 1, 3, 1 }
#define NETWORK_ENABLE_OFFSETS     1
#define TEST_PASS_GRADE            0.05

// #define TRAIN_AND_RUN
 #define TRAIN
 #define RUN
 
// N E U R A L

// Point set for training
std::vector<double> get_train_set() {
	std::vector<double> points(TRAIN_SET_SIZE);
	
	double step = (FUNCTION_END - FUNCTION_START) / static_cast<double>(TRAIN_SET_SIZE + 1);
	for (int i = 0; i < TRAIN_SET_SIZE; ++i)
		points[i] = ((double) i) * step;
	
	return points;
};

// Point set for testring
std::vector<double> get_test_set() {
	std::vector<double> points(TEST_SET_SIZE);
	
	double step = (FUNCTION_END - FUNCTION_START) / static_cast<double>(TEST_SET_SIZE + 1);
	for (int i = 0; i < TEST_SET_SIZE; ++i)
		points[i] = ((double) i) * step + step * 0.3;
	
	return points;
};

// Point set for accurete testing, differs from test set, smaller
std::vector<double> get_accurate_set() {
	std::vector<double> points(ACCURATE_SET_SIZE);
	
	double step = (FUNCTION_END - FUNCTION_START) / static_cast<double>(ACCURATE_SET_SIZE + 1);
	for (int i = 0; i < ACCURATE_SET_SIZE; ++i)
		points[i] = ((double) i) * step + step * 0.6;
	
	return points;
};

// Tested function definition
constexpr double get_function(double t) {
	return FUNCTION_FUNC; // -1.0 + 2.0 * t; // sin(t * 3.141528); // std::sin(2.0 * 3.141528 * t) * std::cos(5.0 * 3.141528 * t);
};

// Perform train of the network on function approximaion
void train_and_run_main() {
	
	NNSpace::MLNetwork network(NETWORK_TOPO);
	network.initialize(INIT_ALGO_ID);
	network.setEnableOffsets(NETWORK_ENABLE_OFFSETS);
	network.setActivator(new NNSpace::NETWORK_ACTIVATOR_FUNCTION());
	
	std::vector<double> set = get_train_set();
	
	std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(set.begin(), set.end(), g);
	
	double rate = START_TRAIN_RATE;
	for (int i = 0; i < TRAIN_SET_SIZE; ++i) {
		std::cout << "Train " << i << " / " << TRAIN_SET_SIZE << std::endl;
		std::vector<double> input = { set[i] };
		std::vector<double> output = { get_function(set[i]) };
		
		rate = std::fabs(network.train_error(TRAIN_ALGO_ID, input, output, rate)) * SCALE_TRAIN_RATE;
		rate = rate <= START_TRAIN_RATE ? START_TRAIN_RATE : rate;
	}
	
	int passed_amount = 0;
	double avg_sqr_error = 0;
	
	std::vector<double> test_set = get_test_set();
	
	for (int i = 0; i < TEST_SET_SIZE; ++i) {
		std::vector<double> input = { test_set[i] };
		std::vector<double> output = network.run(input);
		
		std::cout << "EXPECT: " << get_function(test_set[i]) << ", GOT: " << output[0] << ", ";
		
		avg_sqr_error += (get_function(test_set[i]) - output[0]) * (get_function(test_set[i]) - output[0]) / (double) (TEST_SET_SIZE);
		
		if (std::fabs(get_function(test_set[i]) - output[0]) <= TEST_PASS_GRADE) {
			++passed_amount;
			std::cout << "PASS" << std::endl;
		} else
			std::cout << "MISS : " << (std::fabs(get_function(test_set[i]) - output[0])) << std::endl;
	}
	
	std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
	std::cout.precision(2);
	std::cout << "RESULT: " << passed_amount << '/' << TEST_SET_SIZE << " [" << (100.0 * (double) passed_amount / (double) TEST_SET_SIZE) << "%] avg: " << avg_sqr_error << std::endl;
};

// Perform train of the network & serialize
void train_and_serialize() {
	
	NNSpace::MLNetwork network(NETWORK_TOPO);
	network.initialize(INIT_ALGO_ID);
	network.setEnableOffsets(NETWORK_ENABLE_OFFSETS);
	network.setActivator(new NNSpace::NETWORK_ACTIVATOR_FUNCTION());
	
	std::vector<double> set = get_train_set();
	
	std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(set.begin(), set.end(), g);
	
	double rate = START_TRAIN_RATE;
	for (int i = 0; i < TRAIN_SET_SIZE; ++i) {
		std::cout << "Train " << i << " / " << TRAIN_SET_SIZE << std::endl;
		std::vector<double> input = { set[i] };
		std::vector<double> output = { get_function(set[i]) };
		
		rate = std::fabs(network.train_error(TRAIN_ALGO_ID, input, output, rate)) * SCALE_TRAIN_RATE;
		rate = rate <= START_TRAIN_RATE ? START_TRAIN_RATE : rate;
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
	
	NNSpace::MLNetwork network(NETWORK_TOPO);
	
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
	double avg_sqr_error = 0;
	
	std::vector<double> test_set = get_test_set();
	
	for (int i = 0; i < TEST_SET_SIZE; ++i) {
		std::vector<double> input = { test_set[i] };
		std::vector<double> output = network.run(input);
		
		std::cout << "EXPECT: " << get_function(test_set[i]) << ", GOT: " << output[0] << ", ";
		
		avg_sqr_error += (get_function(test_set[i]) - output[0]) * (get_function(test_set[i]) - output[0]) / (double) (TEST_SET_SIZE);
		
		if (std::fabs(get_function(test_set[i]) - output[0]) <= TEST_PASS_GRADE) {
			++passed_amount;
			std::cout << "PASS" << std::endl;
		} else
			std::cout << "MISS : " << (std::fabs(get_function(test_set[i]) - output[0])) << std::endl;
	}
	
	std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
	std::cout.precision(2);
	std::cout << "RESULT: " << passed_amount << '/' << TEST_SET_SIZE << " [" << (100.0 * (double) passed_amount / (double) TEST_SET_SIZE) << "%] avg: " << avg_sqr_error << std::endl;
};

// bash c.sh "-O3" src/approx_multilayer

int main(int argc, char** argv) {

#ifdef TRAIN_AND_RUN
	train_and_run_main();
#else
#ifdef TRAIN
	train_and_serialize();
#endif
#ifdef RUN
	deserialize_and_run();
#endif
#endif

	return 0;
};