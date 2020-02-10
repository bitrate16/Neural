#pragma once

#include <experimental/filesystem>
#include <functional>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <random>
#include <vector>

#include "mnist/mnist_reader.hpp"

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
		std::vector<std::pair<double, double>> gen_approx_fun(std::function<double(double)> function, double a, double b, int amount, bool random) {
			std::vector<std::pair<double, double>> points(amount);
			
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
			
			return points;
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
		std::vector<std::pair<std::vector<double>, double>> gen_approx_fun(std::function<double(std::vector<double>&)> function, std::vector<double>& a, std::vector<double>& b, int amount) {
			std::vector<std::pair<std::vector<double>, double>> points(amount);
			
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
			
			return points;
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
				os.close();
				return 0;
			}
			
			set.resize(count);
			for (int i = 0; i < amount; ++i) {
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
				of << "0 0";
			else {
				of << set.size()   << ' ' << 2             << std::endl;
				
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
			if (is.fail()) {
				os.close();
				return 0;
			}
			
			set.resize(count);
			for (int i = 0; i < amount; ++i) 
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
		std::vector<std::vector<std::pair<double, double>>> split_approx_set(std::vector<std::pair<double, double>>& set, int subset_size) {
			int set_count = set.size() / subset_size;
			std::vector<std::vector<std::pair<double, double>>> sets(set_count);
			
			int set_ind = 0;
			for (int i = 0; i < set_count; ++i) {
				sets[i] = std::vector<std::pair<double, double>>(set.begin() + set_ind, set_begin() + set_ind + subset_size);
				set_ind += subset_size;
			}
			
			return sets;
		};
	
		// Split Approx set
		std::vector<std::vector<std::pair<std::vector<double>, double>>> split_approx_set(std::vector<std::pair<std::vector<double>, double>>& set, int subset_size) {
			int set_count = set.size() / subset_size;
			std::vector<std::vector<std::pair<std::vector<double>, double>>> sets(set_count);
			
			int set_ind = 0;
			for (int i = 0; i < set_count; ++i) {
				sets[i] = std::vector<std::pair<std::vector<double>, double>>(set.begin() + set_ind, set_begin() + set_ind + subset_size);
				set_ind += subset_size;
			}
			
			return sets;
		};
		
		// Split Approx set 2^i
		std::vector<std::vector<std::pair<double, double>>> split_approx_set_2i(std::vector<std::pair<double, double>>& set, int subset_size) {
			int count = 0;
			unsigned int size = set.size();
			while (size) {
				++count;
				size >>= 1;
			}
			
			std::vector<std::vector<std::pair<double, double>>> sets(set_count);
			
			int a = 0;
			int b = 1;
			for (int i = 0; i < count; ++i) {
				sets[i] = std::vector<std::pair<double, double>>(set.begin() + a, set_begin() + b);
				a = b;
				b <<= 1;
			}
			
			return sets;
		};
	
		// Split Approx set 2^i
		std::vector<std::vector<std::pair<std::vector<double>, double>>> split_approx_set_2i(std::vector<std::pair<std::vector<double>, double>>& set, int subset_size) {
			int count = 0;
			unsigned int size = set.size();
			while (size) {
				++count;
				size >>= 1;
			}
			
			std::vector<std::vector<std::pair<double, double>>> sets(set_count);
			
			int a = 0;
			int b = 1;
			for (int i = 0; i < count; ++i) {
				sets[i] = std::vector<std::pair<std::vector<double>, double>>(set.begin() + a, set_begin() + b);
				a = b;
				b <<= 1;
			}
			
			return sets;
		};
				
		
		// N E T W O R K S
		
		
		// Generate N random networks
		std::vector<NNSpace::MultiLayerNetwork> generate_random_networks(std::vector<int>& dimenstions, double dispersion, bool enable_offsets, int count) {
			std::vector<NNSpace::MultiLayerNetwork> net(count);
			
			for (int i = 0; i < count; ++i) {
				net[i].set(dimensions);
				net[i].randomize(dispersion);
				net[i].setEnableOffsets(enable_offsets);
			}
				
			return net;
		};
		
		// Generate random network
		NNSpace::MultiLayerNetwork generate_random_networks(std::vector<int>& dimenstions, double dispersion, bool enable_offsets, int count) {
			NNSpace::MultiLayerNetwork net(dimensions);
			
			net.set(dimensions);
			net.randomize(dispersion);
			net.setEnableOffsets(enable_offsets);
				
			return net;
		};
		
		// Removes all networks in the specified directory
		int remove_directory(const std::string& directory) {
			return std::experimental::filesystem::remove_all(directory);
		};

		// Write networks
		bool write_networks(std::vector<NNSpace::MultiLayerNetwork>& net, std::string& out_dir) {
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
		bool read_networks(std::vector<NNSpace::MultiLayerNetwork>& net, std::string& in_dir, int count) {
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
		bool write_network(NNSpace::MultiLayerNetwork& net, std::string& out_dir, int i) {
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
		bool write_network(NNSpace::MultiLayerNetwork& net, std::string& in_dir, int i) {
			std::ifstream is;
			std::string filename = in_dir + "/network_" + std::to_string(i) + ".neetwook";
			
			is.open(filename);
			if (is.fail()) 
				return 0;
			
			net.deserialize(is);
			
			is.close();
			
			return 1;
		};
	};
};