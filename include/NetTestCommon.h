/*
 * (c) Copyright bitrate16 (GPLv3.0) 2020
 */

#pragma once

#include <experimental/filesystem>
#include <functional>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <random>
#include <vector>
#include <cmath>

#include "mnist/mnist_reader.hpp"
#include "SingleLayerNetwork.h"
#include "MultiLayerNetwork.h"

// > Appriximation testing
// 1. Generate test for function
// 2. Generate teach for function
// 3. Read set
// 4. Write set

// > Approximate surface
// 1. Generate test for surface
// 2. Generate teach for surface
// 3. Read set
// 4. Write set

// > MNIST
// 1. Read set

// > Set operations
// 1. Split into equal sizes
// 2. Split into 2^i
// 3. Shuffle

// > Network operations
// 1. Generate N random
// 2. Generate 1 random
// 3. Read N
// 4. Write N
// 5. Read 1
// 6. Write 1

namespace NNSpace {
	namespace Common {
		
		
		// A P P R O X I M A T I O N
		
		
		// Generate Set for function approximation
		// function - value generator
		// a        - begin point
		// b        - end point
		// amount   - amount of points in a set
		// random   - points are selected randomly, else sector is split into $amount points
		// Returns std::vector of
		//  std::pair of
		//   double
		//   double
		void gen_approx_fun(std::vector<std::pair<double, double>>& points, std::function<double(double)> function, double a, double b, int amount, bool random) {
			points.resize(amount);
			
			if (random) {
				std::random_device rd;
				std::mt19937 e2(rd());
				std::uniform_real_distribution<> dist(a, b);
				
				for (int i = 0; i < amount; ++i) {
					double p = dist(e2);
					points[i] = std::pair<double, double>(p, function(p));
				}
			} else {		
				double step = (b - a) / static_cast<double>(amount + 1);
				for (int i = 0; i < amount; ++i)
					points[i] = std::pair<double, double>(((double) i) * step, function(((double) i) * step));
			}
		};
		
		
		// Generate Set for ND function approximation, random
		// X value distributed between A and B (Ai <= Xi <= Bi)
		// Assume $function takes as much arguments as size of A and B
		// function - value generator
		// a        - begin point set
		// b        - end point set
		// amount   - amount of points
		// Returns std::vector of
		//  std::pair of
		//   std::vector of double
		//   double
		void gen_approx_fun(std::vector<std::pair<std::vector<double>, double>>& points, std::function<double(std::vector<double>&)> function, std::vector<double>& a, std::vector<double>& b, int amount) {
			points.resize(amount);
			
			for (int i = 0; i < amount; ++i) 
				points[i] = std::pair<std::vector<double>, double>(std::vector<double>(a.size()), 0);
			
			// Create random
			std::random_device rd;
			std::mt19937 e2(rd());
			std::vector<std::uniform_real_distribution<>> dist;
			for (int k = 0; k < a.size(); ++k)
				dist.push_back(std::uniform_real_distribution<>(a[k], b[k]));
			
			for (int i = 0; i < amount; ++i) {
				// Create ND point
				for (int k = 0; k < a.size(); ++k)
					points[i].first[k] = dist[k](e2);
				
				points[i].second = function(points[i].first);
			}
		};
		
		// Write given set to the file
		// set  - input set
		// name - output file name
		bool write_approx_set(std::vector<std::pair<std::vector<double>, double>>& set, const std::string& filename) {
			std::ofstream of;
			of.open(filename);
			if (of.fail()) 
				return 0;
			
			// File structure:
			// Count X_size 
			// Data:{X, Y}
			
			if (set.size() == 0)
				of << "0 0";
			else {
				of << set.size() << ' ' << set[0].first.size() << std::endl;
				
				for (int i = 0; i < set.size(); ++i) {
					for (int k = 0; k < set[i].first.size(); ++k)
						of << set[i].first[k] << ' ';
					
					of << set[i].second << std::endl;
				}
			}
			of.flush();
			of.close();
			
			return 1;
		};
		
		// Read given set from file
		// set  - output set
		// name - input file
		bool read_approx_set(std::vector<std::pair<std::vector<double>, double>>& set, const std::string& filename) {
			std::ifstream is;
			is.open(filename);
			if (is.fail()) 
				return 0;
			
			// File structure:
			// Count X_size 
			// Data:{X, Y}
			
			int count  = 0;
			int x_size = 0;
			
			is >> count >> x_size;
			if (is.fail()) {
				is.close();
				return 0;
			}
			
			set.resize(count);
			for (int i = 0; i < count; ++i) {
				set[i].first.resize(x_size);
				
				for (int k = 0; k < x_size; ++k)
					is >> set[i].first[k];
				
				is >> set[i].second;
			}
			
			return 1;
		};
		
		// Write given set to the file
		// set  - input set
		// name - output file name
		bool write_approx_set(std::vector<std::pair<double, double>>& set, const std::string& filename) {
			std::ofstream of;
			of.open(filename);
			if (of.fail()) 
				return 0;
			
			// File structure:
			// Count X_size 
			// Data:{X, Y}
			
			if (set.size() == 0)
				of << "0 1";
			else {
				of << set.size() << ' ' << 1 << std::endl;
				
				for (int i = 0; i < set.size(); ++i)
					of << set[i].first << ' ' << set[i].second << std::endl;
			}
			of.flush();
			of.close();
			
			return 1;
		};
		
		// Read given set from file
		// set  - output set
		// name - input file
		bool read_approx_set(std::vector<std::pair<double, double>>& set, const std::string& filename) {
			std::ifstream is;
			is.open(filename);
			if (is.fail()) 
				return 0;
			
			// File structure:
			// Count X_size 
			// Data:{X, Y}
			
			int count  = 0;
			int x_size = 0;
			
			is >> count >> x_size;
			
			if (count && x_size != 1)
				return 0;
			
			if (is.fail()) {
				is.close();
				return 0;
			}
			
			set.resize(count);
			for (int i = 0; i < count; ++i) 
				is >> set[i].first >> set[i].second;
			
			return 1;
		};
	
		// Randomly shuffle testing set
		void shuffle_approx_set(std::vector<std::pair<double, double>>& set) {
			std::random_device rd;
			std::mt19937 g(rd());
			std::shuffle(set.begin(), set.end(), g);
		}
		
		// Randomly shuffle testing set
		void shuffle_approx_set(std::vector<std::pair<std::vector<double>, double>>& set) {
			std::random_device rd;
			std::mt19937 g(rd());
			std::shuffle(set.begin(), set.end(), g);
		}
	
		// Split Approx set
		void split_approx_set(std::vector<std::vector<std::pair<double, double>>>& sets, std::vector<std::pair<double, double>>& set, int subset_size, bool append_last = true) {
			int set_count = set.size() / subset_size;
			sets.resize(set_count);
			
			int set_ind = 0;
			for (int i = 0; i < set_count; ++i) {
				sets[i] = std::vector<std::pair<double, double>>(set.begin() + set_ind, set.begin() + set_ind + subset_size);
				set_ind += subset_size;
			}
			if (set_count * subset_size < set.size())
				sets.back().insert(sets.back().end(), set.begin() + set_count * subset_size, set.end());
		};
	
		// Split Approx set
		void split_approx_set(std::vector<std::vector<std::pair<std::vector<double>, double>>>& sets, std::vector<std::pair<std::vector<double>, double>>& set, int subset_size, bool append_last = true) {
			int set_count = set.size() / subset_size;
			sets.resize(set_count);
			
			int set_ind = 0;
			for (int i = 0; i < set_count; ++i) {
				sets[i] = std::vector<std::pair<std::vector<double>, double>>(set.begin() + set_ind, set.begin() + set_ind + subset_size);
				set_ind += subset_size;
			}
			if (set_count * subset_size < set.size())
				sets.back().insert(sets.back().end(), set.begin() + set_count * subset_size, set.end());
		};
		
		// Split Approx set 2^i
		void split_approx_set_2i(std::vector<std::vector<std::pair<double, double>>>& sets, std::vector<std::pair<double, double>>& set) {
			int count = 0;
			unsigned int size = set.size();
			while (size >>= 1) 
				++count;
			
			sets.resize(count);
			
			int a = 0;
			int b = 1;
			for (int i = 0; i < count; ++i) {
				sets[i] = std::vector<std::pair<double, double>>(set.begin() + a, set.begin() + b);
				a = b;
				b <<= 1;
			}
		};
	
		// Split Approx set 2^i
		void split_approx_set_2i(std::vector<std::vector<std::pair<std::vector<double>, double>>>& sets, std::vector<std::pair<std::vector<double>, double>>& set) {
			int count = 0;
			unsigned int size = set.size();
			while (size >>= 1) 
				++count;
			
			sets.resize(count);
			
			int a = 0;
			int b = 1;
			for (int i = 0; i < count; ++i) {
				sets[i] = std::vector<std::pair<std::vector<double>, double>>(set.begin() + a, set.begin() + b);
				a = b;
				b <<= 1;
			}
		};
				
		
		// N E T W O R K S
		
		
		// Generate N random networks
		void generate_random_networks(std::vector<NNSpace::MLNet>& net, std::vector<int>& dimensions, double dispersion, bool enable_offsets, int count) {
			net.resize(count);
			
			for (int i = 0; i < count; ++i) {
				net[i].set(dimensions);
				net[i].randomize(dispersion);
				net[i].setEnableOffsets(enable_offsets);
			}
		};
		
		// Generate random network
		void generate_random_network(NNSpace::MLNet& net, std::vector<int>& dimensions, double dispersion, bool enable_offsets) {
			net.set(dimensions);
			net.randomize(dispersion);
			net.setEnableOffsets(enable_offsets);
		};
		
		// Removes all networks in the specified directory
		int remove_directory(const std::string& directory) {
			return std::experimental::filesystem::remove_all(directory);
		};

		// Write networks
		bool write_networks(std::vector<NNSpace::MLNet>& net, const std::string& out_dir) {
			std::error_code ec;
			if (!std::experimental::filesystem::create_directories(out_dir, ec) && ec)
				return 0;
			
			for (int i = 0; i < net.size(); ++i) {
				std::string filename = out_dir + "/network_" + std::to_string(i) + ".neetwook";
			
				std::ofstream of;
				of.open(filename);
				if (of.fail()) 
					return 0;
				
				net[i].serialize(of);
				
				of.flush();
				of.close();
			}
			
			return 1;
		};
		
		// Read networks
		bool read_networks(std::vector<NNSpace::MLNet>& net, const std::string& in_dir, int count) {
			net.resize(count);
			
			for (int i = 0; i < count; ++i) {
				std::ifstream is;
				std::string filename = in_dir + "/network_" + std::to_string(i) + ".neetwook";
				
				is.open(filename);
				if (is.fail()) 
					return 0;
				
				net[i].deserialize(is);
				
				is.close();
			}
			
			return 1;
		};
		
		// Write single ordered network
		bool write_network(NNSpace::MLNet& net, const std::string& out_dir, int i) {
			std::error_code ec;
			if (!std::experimental::filesystem::create_directories(out_dir, ec) && ec)
				return 0;
			
			std::string filename = out_dir + "/network_" + std::to_string(i) + ".neetwook";
		
			std::ofstream of;
			of.open(filename);
			if (of.fail()) 
				return 0;
			
			net.serialize(of);
			
			of.flush();
			of.close();
			
			return 1;
		};
		
		// Read single ordered network
		bool read_network(NNSpace::MLNet& net, const std::string& in_dir, int i) {
			std::ifstream is;
			std::string filename = in_dir + "/network_" + std::to_string(i) + ".neetwook";
			
			is.open(filename);
			if (is.fail()) 
				return 0;
			
			net.deserialize(is);
			
			is.close();
			
			return 1;
		};
	
		// Write single network
		bool write_network(NNSpace::MLNet& net, const std::string& out_file) {
			std::ofstream of;
			of.open(out_file);
			
			if (of.fail()) 
				return 0;
			
			net.serialize(of);
			
			of.flush();
			of.close();
			
			return 1;
		};
		
		// Read single network
		bool read_network(NNSpace::MLNet& net, const std::string& in_file) {
			std::ifstream is;
			is.open(in_file);
			
			if (is.fail()) 
				return 0;
			
			net.deserialize(is);
			
			is.close();
			
			return 1;
		};
	
		
		// T E S T I N G
		
		
		// Calculate average error on the output layer
		// Ltype defines the L1 or L2 usage.
		double calculate_approx_error(NNSpace::MLNet& net, std::vector<std::pair<double, double>>& set, int Ltype = 1) {
			if (set.size() == 0)
				return 0;
			
			std::vector<double> input(1);
			std::vector<double> output(1);
			
			long double error = 0;
			
			for (int i = 0; i < set.size(); ++i) {
				input[0] = set[i].first;
				net.run(input, output);
				
				long double dv = set[i].second - output[0];
				
				if (Ltype == 1)
					error += std::fabs(dv);
				if (Ltype == 2)
					error += dv * dv;
			}
			
			if (Ltype == 1)
				return error / (double) set.size();
			if (Ltype == 2)
				return std::sqrt(error / (double) set.size());
		};
		
		
		// Calculate average error on the output layer
		// Ltype defines the L1 or L2 usage.
		double calculate_approx_error(NNSpace::MLNet& net, std::vector<std::pair<std::vector<double>, double>>& set, int Ltype = 1) {
			if (set.size() == 0)
				return 0;
			
			std::vector<double> output(1);
			
			long double error = 0;
			
			for (int i = 0; i < set.size(); ++i) {
				net.run(set[i].first, output);
				
				long double dv = set[i].second - output[0];
				if (Ltype == 1)
					error += std::fabs(dv);
				if (Ltype == 2)
					error += dv * dv;
			}
			
			if (Ltype == 1)
				return error / (double) set.size();
			if (Ltype == 2)
				return std::sqrt(error / (double) set.size());
		};
	
		// Calculate max error on the output layer
		long double calculate_approx_error_max(NNSpace::MLNet& net, std::vector<std::pair<double, double>>& set, int Ltype = 1) {
			if (set.size() == 0)
				return 0;
			
			std::vector<double> input(1);
			std::vector<double> output(1);
			
			long double error_max = 0;
			
			for (int i = 0; i < set.size(); ++i) {
				input[0] = set[i].first;
				net.run(input, output);
				
				long double dv = std::fabs(set[i].second - output[0]);
				if (Ltype == 2)
					dv *= dv;
				if (error_max < dv)
					error_max = dv;
			}
			
			return error_max;
		};
		
		
		// Calculate max error on the output layer
		long double calculate_approx_error_max(NNSpace::MLNet& net, std::vector<std::pair<std::vector<double>, double>>& set, int Ltype = 1) {
			if (set.size() == 0)
				return 0;
			
			std::vector<double> output(1);
			
			long double error_max = 0;
			
			for (int i = 0; i < set.size(); ++i) {
				net.run(set[i].first, output);
				
				long double dv = std::fabs(set[i].second - output[0]);
				if (Ltype == 2)
					dv *= dv;
				if (error_max < dv)
					error_max = dv;
			}
			
			return error_max;
		};
	
	
		// M N I S T
		
		
		bool load_mnist(mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t>& set, const std::string& dir) {
			set = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(dir);
			
			return set.training_images.size();
		};
		
		long double calculate_mnist_error(NNSpace::MLNet& net, mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t>& set, int Ltype = 1, int offset = 0, int size = -1) {
			if (size == -1)
				size = set.test_images.size();
			if (offset >= set.test_images.size())
				return 0;
			if (offset + size > set.test_images.size())
				size = set.test_images.size() - offset;
			
			// ???: N size set, M output, how to calculate error?
			
			// Currently: 
			//  L1: SUM [SUM [ABS (dv)] / M] / N
			//  L2: SQRT (SUM [SQRT (SUM [dv ^ 2] / M)] / N)
			
			long double error = 0;
			
			std::vector<double> input(28 * 28);
			std::vector<double> output(10);
			
			for (int i = offset; i < offset + size; ++i) {
				for (int k = 0; k < 28 * 28; ++k)
					input[k] = (double) set.test_images[i][k]  * (1.0 / 255.0);
				
				output = net.run(input);
				
				long double local_error = 0;
				for (int j = 0; j < 10; ++j) {
					long double dv = (set.test_labels[i] == j) ? 1.0 - output[j] : output[j];
					
					if (Ltype == 1)
						local_error += std::fabs(dv);
					else if (Ltype == 2)
						local_error += dv * dv;
				}
				
				if (Ltype == 1)
					error += local_error * 0.1;
				else if (Ltype == 2)
					error += std::sqrt(local_error * 0.1);
			}
			
			if (Ltype == 1)
				return error / (long double) size;
			else if (Ltype == 2)
				return std::sqrt(error / (long double) size);
			return 0;
		};
		
		long double calculate_mnist_match(NNSpace::MLNet& net, mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t>& set, int offset = 0, int size = -1) {
			if (size == -1)
				size = set.test_images.size();
			if (offset >= set.test_images.size())
				return 0;
			if (offset + size > set.test_images.size())
				size = set.test_images.size() - offset;
			
			// Calculate as AmountOfCorrect / Amount
			
			int correct = 0;
			
			std::vector<double> input(28 * 28);
			std::vector<double> output(10);
			
			for (int i = offset; i < offset + size; ++i) {
				for (int k = 0; k < 28 * 28; ++k)
					input[k] = (double) set.test_images[i][k]  * (1.0 / 255.0);
				
				output = net.run(input);
				
				double max = 0;
				double max_ind = 0;
				
				for (int j = 0; j < 10; ++j) 
					if (output[j] > max) {
						max = output[j];
						max_ind = j;
					}
				
				if (max_ind == set.test_labels[i])
					++correct;
			}
			
			return (double) correct / (double) size;
		};
		
		long double calculate_mnist_error_max(NNSpace::MLNet& net, mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t>& set, int Ltype = 1, int offset = 0, int size = -1) {
			if (size == -1)
				size = set.test_images.size();
			if (offset >= set.test_images.size())
				return 0;
			if (offset + size > set.test_images.size())
				size = set.test_images.size() - offset;
			
			long double max_error = 0;
			
			std::vector<double> input(28 * 28);
			std::vector<double> output(10);
			
			for (int i = offset; i < offset + size; ++i) {
				for (int k = 0; k < 28 * 28; ++k)
					input[k] = (double) set.test_images[i][k]  * (1.0 / 255.0);
				
				output = net.run(input);
				
				long double local_error = 0;
				for (int j = 0; j < 10; ++j) {
					long double dv = (set.test_labels[i] == j) ? 1.0 - output[j] : output[j];
					
					if (Ltype == 1)
						local_error += std::fabs(dv);
					else if (Ltype == 2)
						local_error += dv * dv;
				}
				
				if (Ltype == 1)
					local_error = local_error * 0.1;
				else if (Ltype == 2)
					local_error = std::sqrt(local_error * 0.1);
				
				if (max_error < local_error)
					max_error = local_error;
			}
			
			return max_error;
		};
	};
};