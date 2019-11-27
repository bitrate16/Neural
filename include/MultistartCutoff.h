#pragma once

#include <vector>
#include <iostream>
#include <functional>
#include <fstream>
#include <string>
#include <algorithm>
#include <random>
#include "Network.h"
#include "MultiLayerNetwork.h"

namespace NNSpace {
	// Represents single point of linear test set for Y = F(X).
	struct linear_set_point { 
		double x; 
		double y; 
	};
	
	// Generates set normally distributed (if set randomize = 1).
	// Generating linear set on funciton Y = F(X).
	// start <= X <= end,
	// dX = (end - start) / amount.
	void generate_linear_set(std::function<double(double)> function, double start, double end, int amount, const std::string& output_name, bool randomize = 1, bool print = 0) {
		// File structure:
		// <amount of elements>
		// <one by one elements (x, y)>
		
		if (print) {
			std::cout << "[generate_linear_set] Generating " << amount << " samples" << std::endl;
			std::cout << "[generate_linear_set] Generating to " << output_name << std::endl;
		}
		
		std::ofstream of;
		of.open(output_name);
		if (of.fail()) {
			std::cout << "File " << output_name << " open failed" << std::endl;
			return;
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
	};
	
	// Reads linear test set from file.
	// Y = F(X).
	// if randomize = 1, does reflushing of the set.
	// Returns 1 on success, 0 on failture.
	bool read_linear_set(std::vector<linear_set_point>& set, const std::string& input_name, bool randomize = 1, bool print = 0) {
		std::fstream is;
		is.open(input_name);
		
		if (print) 
			std::cout << "[read_linear_set] Reading " << input_name << std::endl;
		
		int amount;
		is >> amount;
		
		set.resize(amount);
		for (int i = 0; i < amount; ++i)
			is >> set[i].x >> set[i].y;
		
		if (is.fail()) {
			is.close();
			return 0;
		}
		
		is.close();
		return 1;
	}
	
	// Input: NET topology.
	// Output: Generates n * W networks of given topology & serializes them to given path.
	void generate_networks(const std::vector<int>& dimensions, const std::string& output_directory, bool print = 0) {
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
			network.setEnableOffsets(0);
			
			// ...
			network.initialize(0);
			// ...
			
			std::string filename = output_directory + "/network_" + std::to_string(i) + ".neetwook";
			
			if (print)
				std::cout << "[generate_network] Generating " << filename << std::endl;
			
			std::ofstream of;
			of.open(filename);
			if (of.fail()) {
				std::cout << "File " << filename << " open failed" << std::endl;
				return;
			}
			network.serialize(of);
			of.flush();
			of.close();
		}
	};
	
	
};