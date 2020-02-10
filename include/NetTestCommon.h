#pragma once

#include <experimental/filesystem>
#include <functional>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <random>
#include <vector>

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
		bool write_approx_set(std::vector<std::pair<double, double>>& set, const std::string& filename) {
			// Create file
			std::error_code ec;
			if (!std::experimental::filesystem::create_directories(output_directory, ec) && ec)
				return 0;

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
					
					of << std::endl << set[i].first[k] << std::endl;
				}
			}
			of.flush();
			of.close();
		};
	};
};