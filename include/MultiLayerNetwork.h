#pragma once

#include "Network.h"

#include <cstdlib>

namespace NNSpace {
	
	// https://habr.com/ru/post/198268/
	
	class MLNetwork : public Network {
		
	public:
		
		// Weights
		std::vector<std::vector<std::vector<double>>> W;
		// Offsets
		std::vector<std::vector<double>> offsets;
		// Dimensions
		std::vector<int> dimensions;
		
		MLNetwork() : Network() {};
		
		MLNetwork(const std::vector<int>& dim) : Network() {
			set(dim);
		};
		
		void set(const std::vector<int>& dim) {
			dimensions = dim;
			
			W.resize(dim.size() - 1);
			for (int i = 0; i < dim.size() - 1; ++i) {
				W[i].resize(dim[i]);
				for (int j = 0; j < dim[i]; ++j)
					W[i][j].resize(dim[i + 1]);
			}
			
			offsets.resize(dimensions.size() - 1);
			for (int i = 0; i < dimensions.size() - 1; ++i)
				offsets[i].resize(dimensions[i + 1]);
		};
		
		void randomize() {
			double v1_MAX = 2.0 / RAND_MAX;
			
			for (int k = 0; k < dimensions.size() - 1; ++k)
				for (int i = 0; i < dimensions[k]; ++i)
					for (int j = 0; j < dimensions[k + 1]; ++j)
						W[k][i][j] = rand() * v1_MAX - 1.0;		
					
			double v2_MAX = 1.0 / RAND_MAX;

			for (int i = 0; i < dimensions.size() - 1; ++i)
				for (int j = 0; j < dimensions[i + 1]; ++j)
					offsets[i][j] = rand() * v2_MAX;
		};
		
		// Teach using backpropagation
		// Assume input, output_teach size match input, output layer size
		void train(const std::vector<double>& input, const std::vector<double>& output_teach, double rate) {
			std::vector<std::vector<double>> layers; // [0-N]
			layers.resize(dimensions.size());
			layers[0] = input;
			
			std::vector<std::vector<double>> layers_raw; // [1-N]
			layers_raw.resize(dimensions.size() - 1);
			
			// Regular process
			for (int k = 0; k < dimensions.size() - 1; ++k) {
				layers[k + 1].resize(dimensions[k + 1]);
				layers_raw[k].resize(dimensions[k + 1]);
				
				// calculate RAW layer outputs & normalize them
				for (int j = 0; j < dimensions[k + 1]; ++j) {
					layers_raw[k][j] = offsets[k][j];
					
					for (int i = 0; i < dimensions[k]; ++i)
						layers_raw[k][j] += layers[k][i] * W[k][i][j];
					
					// Normalize
					layers[k + 1][j] = activator->process(layers_raw[k][j]);
				}
			}
			
			// Weights correction
			std::vector<std::vector<std::vector<double>>> dW;
			
			dW.resize(dimensions.size() - 1);
			for (int i = 0; i < dimensions.size() - 1; ++i) {
				dW[i].resize(dimensions[i]);
				for (int j = 0; j < dimensions[i]; ++j)
					dW[i][j].resize(dimensions[i + 1]);
			}
			
			// Offsets correction
			std::vector<std::vector<double>> doffset;
			
			doffset.resize(dimensions.size() - 1);
			for (int i = 0; i < dimensions.size() - 1; ++i)
				doffset[i].resize(dimensions[i + 1]);
			
			// Sigmas
			std::vector<std::vector<double>> sigma(dimensions.size() - 1);
			for (int i = 0; i < dimensions.size() - 1; ++i)
				sigma[i].resize(dimensions[i + 1]);
			
			// Calculate sigmas
			for (int i = 0; i < dimensions.back(); ++i) // [(dimensions.size() - 1) - 1] // -2
				sigma.back()[i] = (output_teach[i] - layers.back()[i]) * activator->derivative(layers_raw.back()[i]);
			
			for (int k = (dimensions.size() - 1) - 2; k >= 0; --k) // -3
				for (int i = 0; i < dimensions[k + 1]; ++i) {
					for (int j = 0; j < dimensions[k + 2]; ++j)
						sigma[k][i] += sigma[k + 1][j] * W[k + 1][i][j];
					
					sigma[k][i] *= activator->derivative(layers_raw[k + 1 - 1][i]);
				}
			
			// Calculate weights correction
			for (int k = 0; k < dimensions.size() - 1; ++k)
				for (int i = 0; i < dimensions[k]; ++i)
					for (int j = 0; j < dimensions[k + 1]; ++j)
						dW[k][i][j] = rate * sigma[k][j] * layers[k][i];
					
			// Calculate offset correction
			for (int k = 0; k < dimensions.size() - 1; ++k)
				for (int i = 0; i < dimensions[k + 1]; ++i)
					doffset[k][i] = rate * sigma[k][i];
					
			// Balance weights
			for (int k = 0; k < dimensions.size() - 1; ++k)
				for (int i = 0; i < dimensions[k]; ++i)
					for (int j = 0; j < dimensions[k + 1]; ++j)
						W[k][i][j] += dW[k][i][j];
					
			// Balance offsets
			for (int k = 0; k < dimensions.size() - 1; ++k)
				for (int i = 0; i < dimensions[k + 1]; ++i)
					offsets[k][i] += doffset[k][i];
		};
		
		double train_error(const std::vector<double>& input, const std::vector<double>& output_teach, double rate) {
			double out_error_value = 0.0;
			std::vector<std::vector<double>> layers; // [0-N]
			layers.resize(dimensions.size());
			layers[0] = input;
			
			std::vector<std::vector<double>> layers_raw; // [1-N]
			layers_raw.resize(dimensions.size() - 1);
			
			// Regular process
			for (int k = 0; k < dimensions.size() - 1; ++k) {
				layers[k + 1].resize(dimensions[k + 1]);
				layers_raw[k].resize(dimensions[k + 1]);
				
				// calculate RAW layer outputs & normalize them
				for (int j = 0; j < dimensions[k + 1]; ++j) {
					layers_raw[k][j] = offsets[k][j];
					
					for (int i = 0; i < dimensions[k]; ++i)
						layers_raw[k][j] += layers[k][i] * W[k][i][j];
					
					// Normalize
					layers[k + 1][j] = activator->process(layers_raw[k][j]);
				}
			}
			
			// Weights correction
			std::vector<std::vector<std::vector<double>>> dW;
			
			dW.resize(dimensions.size() - 1);
			for (int i = 0; i < dimensions.size() - 1; ++i) {
				dW[i].resize(dimensions[i]);
				for (int j = 0; j < dimensions[i]; ++j)
					dW[i][j].resize(dimensions[i + 1]);
			}
			
			// Offsets correction
			std::vector<std::vector<double>> doffset;
			
			doffset.resize(dimensions.size() - 1);
			for (int i = 0; i < dimensions.size() - 1; ++i)
				doffset[i].resize(dimensions[i + 1]);
			
			// Sigmas
			std::vector<std::vector<double>> sigma(dimensions.size() - 1);
			for (int i = 0; i < dimensions.size() - 1; ++i)
				sigma[i].resize(dimensions[i + 1]);
			
			// Calculate sigmas
			for (int i = 0; i < dimensions.back(); ++i) { // [(dimensions.size() - 1) - 1] // -2
				double dv = output_teach[i] - layers.back()[i];
				sigma.back()[i] = dv * activator->derivative(layers_raw.back()[i]);
				out_error_value += dv;
			}
			
			for (int k = (dimensions.size() - 1) - 2; k >= 0; --k) // -3
				for (int i = 0; i < dimensions[k + 1]; ++i) {
					for (int j = 0; j < dimensions[k + 2]; ++j)
						sigma[k][i] += sigma[k + 1][j] * W[k + 1][i][j];
					
					sigma[k][i] *= activator->derivative(layers_raw[k + 1 - 1][i]);
				}
			
			// Calculate weights correction
			for (int k = 0; k < dimensions.size() - 1; ++k)
				for (int i = 0; i < dimensions[k]; ++i)
					for (int j = 0; j < dimensions[k + 1]; ++j)
						dW[k][i][j] = rate * sigma[k][j] * layers[k][i];
					
			// Calculate offset correction
			for (int k = 0; k < dimensions.size() - 1; ++k)
				for (int i = 0; i < dimensions[k + 1]; ++i)
					doffset[k][i] = rate * sigma[k][i];
					
			// Balance weights
			for (int k = 0; k < dimensions.size() - 1; ++k)
				for (int i = 0; i < dimensions[k]; ++i)
					for (int j = 0; j < dimensions[k + 1]; ++j)
						W[k][i][j] += dW[k][i][j];
					
			// Balance offsets
			for (int k = 0; k < dimensions.size() - 1; ++k)
				for (int i = 0; i < dimensions[k + 1]; ++i)
					offsets[k][i] += doffset[k][i];
				
			return out_error_value / (double) dimensions.back();
		};
		
		// Assume input size match input layer size
		std::vector<double> run(const std::vector<double>& input) {
			std::vector<std::vector<double>> layers; // [0-N]
			layers.resize(dimensions.size());
			layers[0] = input;
			
			// Regular process
			for (int k = 0; k < dimensions.size() - 1; ++k) {
				layers[k + 1].resize(dimensions[k + 1]);
				
				// calculate RAW layer outputs & normalize them
				for (int j = 0; j < dimensions[k + 1]; ++j) {
					layers[k + 1][j] = offsets[k][j];
					
					for (int i = 0; i < dimensions[k]; ++i)
						layers[k + 1][j] += layers[k][i] * W[k][i][j];
					
					// Normalize
					layers[k + 1][j] = activator->process(layers[k + 1][j]);
				}
			}
			
			
			return layers.back();
		};
		
		void serialize(std::ostream& os) {
			// Format:
			// 1. number of layers
			// 2. size of first layer
			// n+1. size of nth layer
			// n+2. one by one bias matrices
			os << dimensions.size();
			for (int k = 0; k < dimensions.size(); ++k)
				os << ' ' << dimensions[k];
			os << ' ';
			
			for (int k = 0; k < dimensions.size() - 1; ++k)
				for (int i = 0; i < dimensions[k]; ++i)
					for (int j = 0; j < dimensions[k + 1]; ++j) 
						os << W[k][i][j] << ' ';
		};
		
		bool deserialize(std::istream& is) {
			int size;
			is >> size;
			dimensions.resize(size);
			
			for (int k = 0; k < dimensions.size(); ++k)
				is >> dimensions[k];
			
			for (int k = 0; k < dimensions.size() - 1; ++k)
				for (int i = 0; i < dimensions[k]; ++i)
					for (int j = 0; j < dimensions[k + 1]; ++j) 
						is >> W[k][i][j];
					
			return 1;
		};
	};
};