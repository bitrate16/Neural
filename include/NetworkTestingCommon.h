#pragma once

#include <vector>
#include <iostream>
#include <functional>
#include <fstream>
#include <string>
#include <experimental/filesystem>
#include <algorithm>
#include <random>
#include "Network.h"
#include "MultiLayerNetwork.h"
#include "mnist/mnist_reader.hpp"

// Contains util functions used in comparing different learning algorithms.
// Used in comparing speed af learning for Multistart algorithm, 
//  standard backpropagation algorithm and random search with learning algorithm.

namespace NNSpace {
	// Represents single point of linear test set for Y = F(X).
	struct linear_set_point { 
		double x; 
		double y; 
	};
	
	// Represents single composite set entry
	struct compose_pair {
		std::vector<double> x;
		std::vector<double> y;
	};
	
	// Utility to calculate log2
	int log2(long n) {
		if (n < 0)
			return 0; // undefined
		
		int t = 0;
		while (n >>= 1) 
			++t;
		
		return t;
	}
	
	// G E N E R A T E _ S E T
	
	// Generates set normally distributed (if set randomize = 1).
	// Generating linear set on funciton Y = F(X).
	// start <= X <= end,
	// dX = (end - start) / amount.
	bool generate_linear_set(std::function<double(double)> function, double start, double end, int amount, const std::string& output_file, bool randomize = 1, bool print = 0) {
		// File structure:
		// <amount of elements>
		// <one by one elements (x, y)>
		
		if (print) {
			std::cout << "[generate_linear_set] Generating " << amount << " samples" << std::endl;
			std::cout << "[generate_linear_set] Generating to " << output_file << std::endl;
		}
		
		std::error_code ec;
		if (!std::experimental::filesystem::create_directories(std::experimental::filesystem::path(output_file).parent_path(), ec) && ec)
			return 0;
		
		std::ofstream of;
		of.open(output_file);
		if (of.fail()) {
			std::cout << "[generate_linear_set] File " << output_file << " open failed" << std::endl;
			return 0;
		}
		
		of << amount << std::endl;
		
		if (randomize) {
			std::vector<double> set(amount);		

			double step = (end - start) / static_cast<double>(amount + 1);
			for (int i = 0; i < amount; ++i)
				set[i] = ((double) i) * step;

			std::random_device rd;
			std::mt19937 g(rd());
			std::shuffle(set.begin(), set.end(), g);
			
			for (int i = 0; i < amount; ++i)
				of << set[i] << ' ' << function(set[i]) << std::endl;
		} else {		
			double step = (end - start) / static_cast<double>(amount + 1);
			for (int i = 0; i < amount; ++i)
				of << ((double) i) * step << ' ' << function(((double) i) * step) << std::endl;
		}
		
				
		of.flush();
		of.close();
		
		return 1;
	};
	
	// S T O R E _ N E T W O R K
	
	// Store single network to directory
	bool store_network(NNSpace::MLNetwork& network, const std::string& filename, bool print = 0) {
		if (print)
			std::cout << "[store_network] Store to " << filename << std::endl;
		std::ofstream of;
		of.open(filename);
		if (of.fail()) {
			std::cout << "[store_network] File " << filename << " open failed" << std::endl;
			return 0;
		}
		network.serialize(of);
		of.flush();
		of.close();
		
		return 1;
	};
	
	// Restore single network from directory
	bool restore_network(NNSpace::MLNetwork& network, const std::string& filename, bool print = 0) {
		if (print)
			std::cout << "[store_network] Restore from " << filename << std::endl;
		std::ifstream is;
		is.open(filename);
		if (is.fail()) {
			std::cout << "[store_network] File " << filename << " open failed" << std::endl;
			return 0;
		}
		network.deserialize(is);
		is.close();
		
		return 1;
	};
	
	// Restore list of networks from directory
	bool restore_networks(std::vector<NNSpace::MLNetwork>& networks, const std::string& directory, int networks_count, bool print) {
		if (print)
			std::cout << "[restore_networks] Restore from " << directory << std::endl;
		
		for (int i = 0; i < networks_count; ++i) {
			std::ifstream is;
			std::string filename = directory + "/network_" + std::to_string(i) + ".neetwook";
			
			is.open(filename);
			if (is.fail()) {
				std::cout << "[restore_networks] File " << filename << " open failed" << std::endl;
				return 0;
			}
			networks[i].deserialize(is);
			is.close();
		}
	};
	
	// S E T  _  R E A D 
	
	// Reads linear test set from file.
	// Y = F(X).
	// if randomize = 1, does reflushing of the set.
	// Returns 1 on success, 0 on failture.
	bool read_linear_set(std::vector<linear_set_point>& set, const std::string& input_name, bool randomize = 1, bool print = 0) {
		if (print) 
			std::cout << "[read_linear_set] Reading " << input_name << std::endl;
		
		std::ifstream is;
		is.open(input_name);
		if (is.fail()) {
			std::cout << "[read_linear_set] File " << input_name << " open failed" << std::endl;
			return 0;
		}
		
		int amount = 0;
		is >> amount;
		
		set.resize(amount);
		for (int i = 0; i < amount; ++i)
			is >> set[i].x >> set[i].y;
		
		if (randomize) {
			std::random_device rd;
			std::mt19937 g(rd());
			std::shuffle(set.begin(), set.end(), g);
		}
		
		/*
		if (is.fail()) {
			is.close();
			return 0;
		}
		*/
		
		is.close();
		return 1;
	}
	
	// used to read set of sets into vector
	bool read_linear_set_set(std::vector<std::vector<linear_set_point>>& set, const std::string& directory_name, int set_size, bool print = 0) {
		if (print) {
			std::cout << "[read_linear_set_set] Reading " << set_size << " sets" << std::endl;
			std::cout << "[read_linear_set_set] Reading from " << directory_name << std::endl;
		}
		
		set.resize(set_size);
		for (int i = 0; i < set_size; ++i) {
			std::ifstream is;
			std::string input_name = directory_name + "/set_" + std::to_string(i) + ".nse";
			
			is.open(input_name);
			if (is.fail()) {
				std::cout << "[read_linear_set_set] File " << input_name << " open failed" << std::endl;
				return 0;
			}
			
			int amount = 0;
			is >> amount;
			
			set.resize(amount);
			for (int j = 0; j < amount; ++j)
				is >> set[i][j].x >> set[i][j].y;
			
			is.close();
		}
		
		return 1;
	}
	
	// S E T  _ S P L I T
	
	// Reads input set, randomizes and splits it into N pieces, writes each part into passed directory
	bool split_linear_set(const std::string& input_name, const std::string& output_directory, int split_amount = 1, bool randomize_before_split = 1, bool print = 0) {
		if (print) 
			std::cout << "[split_linear_set] Splitting set " << input_name << " into " << split_amount << " subsets to " << output_directory << std::endl;
		
		std::error_code ec;
		if (!std::experimental::filesystem::create_directories(output_directory, ec) && ec)
			return 0;
		
		std::vector<linear_set_point> input_set;
		if (!read_linear_set(input_set, input_name, randomize_before_split, print)) 
			return 0;
		
		// Bound split count
		split_amount = split_amount < 0 ? 1 : split_amount;
		split_amount = split_amount > input_set.size() ? input_set.size() : split_amount;
		
		// Generate splits
		int part_size = input_set.size() / split_amount;
		for (int i = 0; i < split_amount; ++i) {
			std::string output_file = output_directory + "/set_" + std::to_string(i) + ".nse";
			
			if (print)
				std::cout << "[split_linear_set] Split [" << i << "] to " << output_file << std::endl;
			
			std::ofstream of;
			of.open(output_file);
			if (of.fail()) {
				std::cout << "[split_linear_set] File " << output_file << " open failed" << std::endl;
				return 0;
			}
			
			of << part_size << std::endl;
			for (int k = i * part_size; k < (i + 1) * part_size; ++k) 
				of << input_set[k].x << ' ' << input_set[k].y << std::endl;
					
			of.flush();
			of.close();
		}
		
		return 1;
	};
	
	// Reads input set, randomizes and splits it into N pieces, writes each part into passed directory
	void split_linear_set(std::vector<std::vector<linear_set_point>>& set_set, std::vector<linear_set_point>& set, int split_amount = 1, bool randomize_before_split = 1, bool print = 0) {
		if (print) 
			std::cout << "[split_linear_set] Splitting set into " << split_amount << " subsets" << std::endl;
		
		// Bound split count
		split_amount = split_amount < 0 ? 1 : split_amount;
		split_amount = split_amount > set.size() ? set.size() : split_amount;
		set_set.resize(split_amount);
		
		// Generate splits
		int part_size = set.size() / split_amount;
		for (int i = 0; i < split_amount; ++i) {
			set_set[i].resize(part_size);
			set_set[i].assign(&set[i * part_size], &set[(i + 1) * part_size]);
		}
	};
	
	// Used to split set into 1/2, 1/4, 1/8, ...
	void split_linear_set_base_2(std::vector<std::vector<linear_set_point>>& set_set, std::vector<linear_set_point>& set, int A, bool randomize_before_split = 1, bool print = 0) {
		
		// Ci = C1 * 2 ^ (i-1)
		// C1 = C / (2 ^ [log2(A)] - 1)
		
		int C1 = set.size() / ((1 << log2(A)) - 1);
		int Ci = C1;
		int N = log2(A);
		int current = 0;
		int current_size = C1;
		
		if (print) 
			std::cout << "[split_linear_set_base_2] Splitting set, C = " << set.size() << ", C1 = " << C1 << ", N = " << N << ", Cn = " << (Ci << (N - 1)) << std::endl;
		
		// Generate splits
		set_set.resize(N);
		for (int i = 0; i < N; ++i) {
			// std::cout << "Ci = " << Ci << ", current = " << current << ", current_size = " << current_size << std::endl;
			set_set[i].resize(current_size);
			set_set[i].assign(&set[current], &set[current + current_size]);
			int Cj       = C1 << i + 1;
			current     += current_size;
			current_size = Cj;
			Ci           = Cj;
		}
	};
	
	// Split existing MNIST dataset into subsets
	void split_compose_set_base_2(std::vector<std::vector<compose_pair>>& set_set, std::vector<compose_pair>& set, int A, bool randomize_before_split = 1, bool print = 0) {
		
		// Ci = C1 * 2 ^ (i-1)
		// C1 = C / (2 ^ [log2(A)] - 1)
		
		int C1 = set.size() / ((1 << log2(A)) - 1);
		int Ci = C1;
		int N = log2(A);
		int current = 0;
		int current_size = C1;
		
		if (print) 
			std::cout << "[split_linear_set_base_2] Splitting set, C = " << set.size() << ", C1 = " << C1 << ", N = " << N << ", Cn = " << (Ci << (N - 1)) << std::endl;
		
		// Generate splits
		set_set.resize(N);
		for (int i = 0; i < N; ++i) {
			// std::cout << "Ci = " << Ci << ", current = " << current << ", current_size = " << current_size << std::endl;
			set_set[i].resize(current_size);
			set_set[i].assign(&set[current], &set[current + current_size]);
			int Cj       = C1 << i + 1;
			current     += current_size;
			current_size = Cj;
			Ci           = Cj;
		}
	};
	
	// R A N D O M _ N E T W O R K S
	
	// Input: NET topology.
	// Output: Generates n * W networks of given topology & serializes them to given path.
	// Activator function is undefined.
	bool generate_random_weight_networks(const std::vector<int>& dimensions, const std::string& output_directory, ActivatorType activator = ActivatorType::LINEAR, double dispersion = 0, bool enable_offsets = 0, bool print = 0) {
		
		std::error_code ec;
		if (!std::experimental::filesystem::create_directories(output_directory, ec) && ec)
			return 0;
		
		// Number of topos is A = n * W,
		//                    n - count of input layers.
		//                    W - count of weights.
		int num_topos = 0;
		for (int i = 0; i < dimensions.size() - 1; ++i)
			num_topos += dimensions[i] * dimensions[i + 1];
		num_topos *= dimensions[0];
		
		// XXX: Organize algorithm to generate A subsets of weight local points to cover entire topology.
		
		for (int i = 0; i < num_topos; ++i) {
			NNSpace::MLNetwork network(dimensions);
			network.setEnableOffsets(enable_offsets);
			network.setActivator(getActivatorByType(activator));
			
			// ...
			network.initialize(dispersion);
			// ...
			
			std::string filename = output_directory + "/network_" + std::to_string(i) + ".neetwook";
			
			if (print)
				std::cout << "[generate_random_weight_networks] Generating " << filename << std::endl;
			
			std::ofstream of;
			of.open(filename);
			if (of.fail()) {
				std::cout << "[generate_random_weight_networks] File " << filename << " open failed" << std::endl;
				return 0;
			}
			network.serialize(of);
			of.flush();
			of.close();
		}
		
		return 1;
	};
	
	void generate_random_weight_networks(std::vector<NNSpace::MLNetwork>& networks, const std::vector<int>& dimensions, ActivatorType activator = ActivatorType::LINEAR, double dispersion = 0, bool enable_offsets = 0, bool print = 0) {
		// Number of topos is A = n * W,
		//                    n - count of input layers.
		//                    W - count of weights.
		int num_topos = 0;
		for (int i = 0; i < dimensions.size() - 1; ++i)
			num_topos += dimensions[i] * dimensions[i + 1];
		num_topos *= dimensions[0];
		
		// XXX: Organize algorithm to generate A subsets of weight local points to cover entire topology.
		
		networks.resize(num_topos);
		for (int i = 0; i < num_topos; ++i) {
			networks[i] = NNSpace::MLNetwork(dimensions);
			networks[i].setEnableOffsets(enable_offsets);
			networks[i].setActivator(getActivatorByType(activator));
			
			// ...
			networks[i].initialize(dispersion);
			// ...
			
			if (print)
				std::cout << "[generate_random_weight_networks] Generating # " << i << std::endl;
		}
	};
	
	bool generate_random_weight_networks_count(const std::vector<int>& dimensions, const std::string& output_directory, ActivatorType activator = ActivatorType::LINEAR, double dispersion = 0, int net_cnt = 1, bool enable_offsets = 0, bool print = 0) {
		
		std::error_code ec;
		if (!std::experimental::filesystem::create_directories(output_directory, ec) && ec)
			return 0;
		
		// XXX: Organize algorithm to generate A subsets of weight local points to cover entire topology.
		
		for (int i = 0; i < net_cnt; ++i) {
			NNSpace::MLNetwork network(dimensions);
			network.setEnableOffsets(enable_offsets);
			network.setActivator(getActivatorByType(activator));
			
			// ...
			network.initialize(dispersion);
			// ...
			
			std::string filename = output_directory + "/network_" + std::to_string(i) + ".neetwook";
			
			if (print)
				std::cout << "[generate_random_weight_networks] Generating " << filename << std::endl;
			
			std::ofstream of;
			of.open(filename);
			if (of.fail()) {
				std::cout << "[generate_random_weight_networks] File " << filename << " open failed" << std::endl;
				return 0;
			}
			network.serialize(of);
			of.flush();
			of.close();
		}
		
		return 1;
	};
	
	void generate_random_weight_networks_count(std::vector<NNSpace::MLNetwork>& networks, const std::vector<int>& dimensions, ActivatorType activator = ActivatorType::LINEAR, double dispersion = 0, int net_cnt = 1, bool enable_offsets = 0, bool print = 0) {
		
		// XXX: Organize algorithm to generate A subsets of weight local points to cover entire topology.
		
		networks.resize(net_cnt);
		for (int i = 0; i < net_cnt; ++i) {
			networks[i] = NNSpace::MLNetwork(dimensions);
			networks[i].setEnableOffsets(enable_offsets);
			networks[i].setActivator(getActivatorByType(activator));
			
			// ...
			networks[i].initialize(dispersion);
			// ...
			
			if (print)
				std::cout << "[generate_random_weight_networks] Generating # " << i << std::endl;
		}
	};
	
	// Generates single nework, initialize it with passed dispersion value
	void generate_random_weight_network(NNSpace::MLNetwork& network, const std::vector<int>& dimensions, ActivatorType activator = ActivatorType::LINEAR, double dispersion = 0, bool enable_offsets = 0, bool print = 0) {
		network = NNSpace::MLNetwork(dimensions);
		network.setEnableOffsets(enable_offsets);
		network.setActivator(getActivatorByType(activator));
		
		// ...
		network.initialize(dispersion);
		// ...
		
		if (print)
			std::cout << "[generate_random_weight_network] Generating single network" << std::endl;
	};
	
	// R E M O V E 
	
	// Removes all networks in the specified directory
	int remove_directory(const std::string& directory, bool print = 0) {
		if (print)
			std::cout << "[remove_directory] Purge directory " << directory << std::endl;
		
		return std::experimental::filesystem::remove_all(directory);
	};

	// T R A I N I N G

	// Train passed network on passed set.
	void train_network_backpropagation(NNSpace::MLNetwork& network, std::vector<linear_set_point>& train_set, int id = 0, bool print = 0) {
		if (train_set.size() == 0)
			return;
		
		// Start training from calculating error value on the first test like it was continuous train loop.
		double rate = 0.0;
		
		// Calculate next step error
		{
			std::vector<double> input = { train_set[0].x };
			std::vector<double> output = network.run(input);
			
			double dv = output[0] - train_set[0].y;
			if (id == 0)
				rate = std::fabs(dv);
			else if (id == 1)
				rate = std::fabs(dv);
		}
		
		// Keep teaching
		for (int i = 0; i < train_set.size(); ++i) {
			if (print)
				std::cout << "[train_network_backpropagation] Train " << i << " / " << train_set.size() << std::endl;
			
			// XXX: implement other training algorithm
			rate = network.train_error(id, { train_set[i].x }, { train_set[i].y }, rate);
		}
	};

	// Train passed network on passed set.
	void train_compose_network_backpropagation(NNSpace::MLNetwork& network, std::vector<compose_pair>& train_set, int id = 0, bool print = 0) {
		if (train_set.size() == 0)
			return;
		
		// Start training from calculating error value on the first test like it was continuous train loop.
		double rate = 0.0;
		
		// Calculate next step error
		{
			std::vector<double> output = network.run(train_set[0].x);
			
			for (int i = 0; i < train_set[0].y.size(); ++i) {
				double dv = train_set[0].y[i] - output[i];
				
				if (id == 0)
					rate += dv * dv;
				else if (id == 1)
					rate += std::fabs(dv);
			}
			
			rate /= (double) train_set[0].y.size();
		}
		
		// Keep teaching
		for (int i = 0; i < train_set.size(); ++i) {
			if (print)
				std::cout << "[train_compose_network_backpropagation] Train " << i << " / " << train_set.size() << std::endl;
			
			// XXX: implement other training algorithm
			rate = network.train_error(id, train_set[i].x, train_set[i].y, rate);
		}
	};

	// E R R O R _ C A L C U L A T I O N

	// SUM [ e^2 ] / amount
	long double calculate_square_error(NNSpace::MLNetwork& network, std::vector<linear_set_point>& set, bool print = 0) {
		if (print)
			std::cout << "[calculate_square_error] Calculating square error value" << std::endl;
		
		std::vector<double> input(1);
		std::vector<double> output;
		
		long double error = 0;
		
		for (int i = 0; i < set.size(); ++i) {
			input[0] = set[i].x;
			output = network.run(input);
			
			long double dv = set[i].y - output[0];
			error += dv * dv;
		}
		
		return std::sqrt(error / (double) set.size());
	}
	
	// SUM [ |e| ] / amount
	long double calculate_linear_error(NNSpace::MLNetwork& network, std::vector<linear_set_point>& set, bool print = 0) {
		if (print)
			std::cout << "[calculate_linear_error] Calculating linear error value" << std::endl;
		
		std::vector<double> input(1);
		std::vector<double> output;
		
		long double error = 0;
		
		for (int i = 0; i < set.size(); ++i) {
			input[0] = set[i].x;
			output = network.run(input);
			
			long double dv = set[i].y - output[0];
			error += std::fabs(dv);
		}
		
		return error / (double) set.size();
	}

	// SUM [ SUM [ | output[i] - test[i] | ] ^ 2 ] / amount
	long double calculate_compose_square_error(NNSpace::MLNetwork& network, std::vector<compose_pair>& set, bool print = 0) {
		if (print)
			std::cout << "[calculate_square_error] Calculating square error value" << std::endl;
		
		if (set.size() == 0)
			return 0.0;
		
		std::vector<double> output;
		
		long double error = 0;
		
		for (int i = 0; i < set.size(); ++i) {
			output = network.run(set[i].x);
			
			// Error is summary from entire layer
			long double dv = 0.0;
			
			for (int i = 0; i < set[i].x.size(); ++i) 
				dv += std::fabs(set[i].y[i] - output[i]);
			
			error += dv * dv;
		}
		
		return std::sqrt(error / (double) set.size());
	}
	
	// SUM [ SUM [ | output[i] - test[i] | ] ] / amount
	long double calculate_compose_linear_error(NNSpace::MLNetwork& network, std::vector<compose_pair>& set, bool print = 0) {
		if (print)
			std::cout << "[calculate_compose_linear_error] Calculating square error value" << std::endl;
		
		if (set.size() == 0)
			return 0.0;
		
		std::vector<double> output;
		
		long double error = 0;
		
		for (int i = 0; i < set.size(); ++i) {
			output = network.run(set[i].x);
			
			// Error is summary from entire layer
			
			for (int i = 0; i < set[i].x.size(); ++i) 
				error += std::fabs(set[i].y[i] - output[i]);
		}
		
		return error / (double) set.size();
	}

	// - - - - M N I S T - - - -
	
	// Convert MNIST learning dataset to network-acceptable vector of doubles
	void convert_mnist_learn(std::vector<compose_pair>& set, mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t>& dataset, int label_dimensions) {
		set.resize(dataset.training_images.size());
		
		for (int i = 0; i < dataset.training_images.size(); ++i) {
			set[i].x.resize(dataset.training_images[i].size());
			set[i].y.resize(label_dimensions);
			
			for (int j = 0; j < dataset.training_images[i].size(); ++j) 
				set[i].x[j] = (double) dataset.training_images[i][j] / 255.0;
			
			set[i].y[dataset.training_labels[i]] = 1.0;
		}
	};
	
	// SUM [ SUM [ | output[i] - test[i] | ] ^ 2 ] / amount
	// test_size - is amount of testing elements to be used to calculate error.
	// Assuming datased is shuffled.
	// label_size - is amount of neurons encoding output signal.
	//  In case of digits matching it is 10. (0 1 2 3 4 5 6 7 8 9)
	long double calculate_mnist_square_error(NNSpace::MLNetwork& network, mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t>& dataset, int test_size, int label_size) {
		if (test_size < 0)
			test_size = 0;
		if (test_size > dataset.test_images.size())
			test_size = dataset.test_images.size() - 1;
		
		if (test_size == 0)
			return 0.0;
		
		std::vector<double> input(dataset.test_images[0].size());
		std::vector<double> output;
		std::vector<double> test(label_size);
		
		long double error = 0;
		
		for (int i = 0; i < test_size; ++i) {
			// Input
			for (int k = 0; k < dataset.training_images[0].size(); ++k)
				input[k] = (double) dataset.test_images[i][k] / 255.0;
			
			// Test
			test[dataset.test_labels[i]] = 1.0;
			
			// Output
			output = network.run(input);
			
			// Error is summary from entire layer
			long double dv = 0.0;
			
			for (int k = 0; k < test.size(); ++k) 
				dv += std::fabs(test[k] - output[k]);
			
			error += dv * dv;
			
			// Reset test
			test[dataset.test_labels[i]] = 0.0;
		}
		
		return std::sqrt(error / (double) test_size);
	};
	
	// SUM [ SUM [ | output[i] - test[i] | ] ] / amount
	// test_size - is amount of testing elements to be used to calculate error.
	// Assuming datased is shuffled.
	// label_size - is amount of neurons encoding output signal.
	//  In case of digits matching it is 10. (0 1 2 3 4 5 6 7 8 9)
	long double calculate_mnist_linear_error(NNSpace::MLNetwork& network, mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t>& dataset, int test_size, int label_size) {
		if (test_size < 0)
			test_size = 0;
		if (test_size > dataset.test_images.size())
			test_size = dataset.test_images.size() - 1;
		
		if (test_size == 0)
			return 0.0;
		
		std::vector<double> input(dataset.test_images[0].size());
		std::vector<double> output;
		std::vector<double> test(label_size);
		
		long double error = 0;
		
		for (int i = 0; i < test_size; ++i) {
			// Input
			for (int k = 0; k < dataset.test_images[i].size(); ++k)
				input[k] = (double) dataset.test_images[i][k] / 255.0;
			
			// Test
			test[dataset.test_labels[i]] = 1.0;
			
			// Output
			output = network.run(input);
			
			// Error is summary from entire layer
			long double dv = 0.0;
			
			for (int k = 0; k < test.size(); ++k) 
				error += std::fabs(test[k] - output[k]);
			
			// Reset test
			test[dataset.test_labels[i]] = 0.0;
		}
		
		return error / (double) test_size;
	};

	// Train passed network on passed MNIST set.
	void train_mnist_network_backpropagation(NNSpace::MLNetwork& network, mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t>& dataset, int start_index, int train_size, int label_size, int error_calc_id = 0) {
		if (dataset.training_images.size() == 0 || train_size == 0 || start_index + train_size > dataset.training_images.size())
			return;
		
		// Start training from calculating error value on the first test like it was continuous train loop.
		double rate = 0.0;
		
		std::vector<double> input(dataset.training_images[start_index].size());
		std::vector<double> output;
		std::vector<double> test(label_size);
		
		// Calculate next step error
		{
			// Input
			for (int k = 0; k < dataset.training_images[start_index].size(); ++k)
				input[k] = (double) dataset.training_images[start_index][k] / 255.0;
			
			// Test
			test[dataset.training_labels[start_index]] = 1.0;
			
			// Output
			output = network.run(input);
		
			for (int i = 0; i < test.size(); ++i) {
				double dv = test[i] - output[i];
				
				if (error_calc_id == 0)
					rate += dv * dv;
				else if (error_calc_id == 1)
					rate += std::fabs(dv);
			}
			
			rate /= (double) test.size();
			
			if (error_calc_id == 1)
				rate = std::sqrt(rate);
		}
		
		// Keep teaching
		for (int i = start_index; i < start_index + train_size; ++i) {
			if (i != 0) {				
		
				// Input
				for (int k = 0; k < dataset.training_images[i].size(); ++k)
					input[k] = (double) dataset.training_images[i][k] / 255.0;
				
				// Test
				test[dataset.training_labels[i]] = 1.0;
			}
			
			// XXX: implement other training algorithm
			rate = network.train_error(error_calc_id, input, test, rate);
			
			test[dataset.training_labels[i]] = 0.0;
		}
	};

	// amount_of_correct_answers / amount
	// test_size - is amount of testing elements to be used to calculate error.
	// Assuming datased is shuffled.
	// label_size - is amount of neurons encoding output signal.
	//  In case of digits matching it is 10. (0 1 2 3 4 5 6 7 8 9)
	long double calculate_mnist_match_error(NNSpace::MLNetwork& network, mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t>& dataset, int test_size, int label_size) {
		if (test_size < 0)
			test_size = 0;
		if (test_size > dataset.test_images.size())
			test_size = dataset.test_images.size() - 1;
		
		if (test_size == 0)
			return 0.0;
		
		std::vector<double> input(dataset.test_images[0].size());
		std::vector<double> output;
		
		int correct = 0;
		
		for (int i = 0; i < test_size; ++i) {
			// Input
			for (int k = 0; k < dataset.training_images[0].size(); ++k)
				input[k] = (double) dataset.test_images[i][k] / 255.0;
						
			// Output
			output = network.run(input);
			
			int    max_match     = 0;
			double max_match_ind = std::numeric_limits<double>::min();
			
			for (int i = 0; i < label_size; ++i)
				if (output[i] > max_match) {
					max_match = output[i];
					max_match_ind = i;
				}
				
			if (dataset.test_labels[i] == max_match_ind)
				++correct;
		}
		
		return (long double) correct / (long double) test_size;
	};
	

	// Functions without files
};