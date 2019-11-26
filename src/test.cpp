#include "Network.h"
#include "MultiLayerNetwork.h"
#include "SingleLayerNetwork.h"

#include <iostream>
#include <random>
#include <cstdio>
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
	
std::vector<double> set = get_train_set();
double rate = START_TRAIN_RATE;
int step = 0;

NNSpace::SLNetwork network1(1, 3, 1);
NNSpace::MLNetwork network2({1, 3, 1});

// Perform train of the network & serialize
void train_step() {
	if (step < TRAIN_SET_SIZE) {
		// std::cout << "Train " << step << " / " << TRAIN_SET_SIZE << std::endl;
		
		std::vector<double> input = { set[step] };
		std::vector<double> output = { get_function(set[step]) };
		
		std::cout << "NET # " << step << std::endl << std::endl;

		                 network1.train_error(TRAIN_ALGO_ID, input, output, rate);

		std::cerr << "NET # " << step << std::endl << std::endl;
						 
						 
		rate = std::fabs(network2.train_error(TRAIN_ALGO_ID, input, output, rate)) * SCALE_TRAIN_RATE;
		rate = rate <= START_TRAIN_RATE ? START_TRAIN_RATE : rate;
		
		++step;
	}
};

// bash c.sh "-O3" src/test

int main(int argc, char** argv) {
	
	std::cout << "Deserializing SL network from " << NETWORK_SERIALIZE_PATH << std::endl;
	std::ifstream ifs;
	ifs.open(NETWORK_SERIALIZE_PATH);
	if (ifs.fail()) {
		std::cout << "File " << NETWORK_SERIALIZE_PATH << " not found" << std::endl;
		return 1;
	}
	if (!network1.deserialize(ifs)) {
		std::cout << "Deserialize failed" << std::endl;
		ifs.close();
		return 1;
	}
	ifs.close();
	
	std::cout << "Deserializing ML network from " << NETWORK_SERIALIZE_PATH << std::endl;
	ifs.open(NETWORK_SERIALIZE_PATH);
	if (ifs.fail()) {
		std::cout << "File " << NETWORK_SERIALIZE_PATH << " not found" << std::endl;
		return 1;
	}
	if (!network2.deserialize(ifs)) {
		std::cout << "Deserialize failed" << std::endl;
		ifs.close();
		return 1;
	}
	ifs.close();
	
	freopen("netout1.txt","w",stdout);
	freopen("netout2.txt","w",stderr);

	for (int i = 0; i < 10000; ++i)
		train_step();

	return 0;
};