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
		// Activators
		std::vector<NetworkFunction*> activators;
		// Dimensions
		std::vector<int> dimensions;
		
		bool enable_offsets = 0;
		
		MLNetwork() : Network() {};
		
		MLNetwork(const std::vector<int>& dim) : Network() {
			set(dim);
		};
		
		void set(const std::vector<int>& dim) {
			dimensions = dim;
			
			W.clear();
			
			W.resize(dim.size() - 1);
			for (int i = 0; i < dim.size() - 1; ++i) {
				W[i].resize(dim[i]);
				for (int j = 0; j < dim[i]; ++j)
					W[i][j].resize(dim[i + 1]);
			}
			
			offsets.clear();
			
			offsets.resize(dimensions.size() - 1);
			for (int i = 0; i < dimensions.size() - 1; ++i)
				offsets[i].resize(dimensions[i + 1]);
			
			while (activators.size() + 1 > dim.size()) {
				delete activators.back();
				activators.pop_back();
			}
			
			while (activators.size() + 1 < dim.size())
				activators.push_back(new Linear());
		};
		
		void setLayerActivator(int layer, NetworkFunction* function) {
			if (layer > 0 && layer < activators.size() + 1) {
				delete activators[layer];
				activators[layer] = function;
			} else
				throw std::runtime_error("Invalid layer number");
		};
		
		virtual void setActivator(NetworkFunction* function) {
			delete activators[0];
			activators[0] = function;
				
			for (int i = 1; i < activators.size(); ++i) {
				delete activators[i];
				activators[i] = function->clone();
			}
		};
		
		void initialize(double dispersion) {
			double scale2 = dispersion * 0.5;
			double v1_MAX = dispersion / RAND_MAX;
			
			for (int k = 0; k < dimensions.size() - 1; ++k)
				for (int i = 0; i < dimensions[k]; ++i)
					for (int j = 0; j < dimensions[k + 1]; ++j)
						W[k][i][j] = rand() * v1_MAX - scale2;		

			for (int i = 0; i < dimensions.size() - 1; ++i)
				for (int j = 0; j < dimensions[i + 1]; ++j)
					offsets[i][j] = rand() * v1_MAX - scale2;
		};
		
		inline void setEnableOffsets(bool e) {
			enable_offsets = e;
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
					layers_raw[k][j] = enable_offsets ? offsets[k][j] : 0;
					
					for (int i = 0; i < dimensions[k]; ++i)
						layers_raw[k][j] += layers[k][i] * W[k][i][j];
					
					// Normalize
					layers[k + 1][j] = activators[k]->process(layers_raw[k][j]);
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
			std::vector<std::vector<double>> doffset(dimensions.size() - 1);
			// Sigmas
			std::vector<std::vector<double>> sigma(dimensions.size() - 1);
			
			for (int i = 0; i < dimensions.size() - 1; ++i) {
				doffset[i].resize(dimensions[i + 1]);
				sigma[i].resize(dimensions[i + 1]);
			}
			
			// Calculate sigmas
			for (int i = 0; i < dimensions.back(); ++i) { // K-2, K-1
				double dv = output_teach[i] - layers.back()[i];
				sigma.back()[i] = dv * activators.back()->derivative(layers_raw.back()[i]);
			}
			
			for (int k = (dimensions.size() - 1) - 2; k >= 0; --k) // K-3, K-2,, ..
				for (int i = 0; i < dimensions[k + 1]; ++i) {
					for (int j = 0; j < dimensions[k + 2]; ++j)
						sigma[k][i] += sigma[k + 1][j] * W[k + 1][i][j];
					
					sigma[k][i] *= activators[k + 0]->derivative(layers_raw[k + 1 - 1][i]); // layers_raw[k + 1]
				} // checked?
				// XXX: Debug with print sigmas step by step for same seed
			
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
				for (int i = 0; i < dW[k].size(); ++i)
					for (int j = 0; j < dW[k][i].size(); ++j)
						W[k][i][j] += dW[k][i][j];
					
			// Balance offsets
			for (int k = 0; k < dimensions.size() - 1; ++k)
				for (int i = 0; i < dimensions[k + 1]; ++i)
					offsets[k][i] += doffset[k][i];
		};
		
		double train_error(int error_calculate_id, const std::vector<double>& input, const std::vector<double>& output_teach, double rate) {
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
					layers_raw[k][j] = enable_offsets ? offsets[k][j] : 0;
					
					for (int i = 0; i < dimensions[k]; ++i)
						layers_raw[k][j] += layers[k][i] * W[k][i][j];
					
					// Normalize
					layers[k + 1][j] = activators[k]->process(layers_raw[k][j]);
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
			std::vector<std::vector<double>> doffset(dimensions.size() - 1);
			// Sigmas
			std::vector<std::vector<double>> sigma(dimensions.size() - 1);
			
			for (int i = 0; i < dimensions.size() - 1; ++i) {
				doffset[i].resize(dimensions[i + 1]);
				sigma[i].resize(dimensions[i + 1]);
			}
			
			// Calculate sigmas
			for (int i = 0; i < dimensions.back(); ++i) { // K-2, K-1
				double dv = output_teach[i] - layers.back()[i];
				sigma.back()[i] = dv * activators.back()->derivative(layers_raw.back()[i]);
				
				if (error_calculate_id == 0)
					out_error_value += dv * dv;
				else if (error_calculate_id == 1)
					out_error_value += std::fabs(dv);
			}
			
			for (int k = (dimensions.size() - 1) - 2; k >= 0; --k) // K-3, K-2,, ..
				for (int i = 0; i < dimensions[k + 1]; ++i) {
					for (int j = 0; j < dimensions[k + 2]; ++j)
						sigma[k][i] += sigma[k + 1][j] * W[k + 1][i][j];
					
					sigma[k][i] *= activators[k + 0]->derivative(layers_raw[k + 1 - 1][i]); // layers_raw[k + 1]
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
				for (int i = 0; i < dW[k].size(); ++i)
					for (int j = 0; j < dW[k][i].size(); ++j)
						W[k][i][j] += dW[k][i][j];
					
			// Balance offsets
			for (int k = 0; k < dimensions.size() - 1; ++k)
				for (int i = 0; i < dimensions[k + 1]; ++i)
					offsets[k][i] += doffset[k][i];
			/*	
			// PRINT
			std::cerr << "sigmas:" << std::endl;
			for (int k = 0; k < sigma.size(); ++k) {
				for (int i = 0; i < sigma[k].size(); ++i)
					std::cerr << sigma[k][i] << ' ';
				
				std::cerr << std::endl;
			}
			std::cerr << "d offsets:" << std::endl;
			for (int k = 0; k < doffset.size(); ++k) {
				for (int i = 0; i < doffset[k].size(); ++i)
					std::cerr << doffset[k][i] << ' ';
				
				std::cerr << std::endl;
			}
			std::cerr << "offsets:" << std::endl;
			for (int k = 0; k < offsets.size(); ++k) {
				for (int i = 0; i < offsets[k].size(); ++i)
					std::cerr << offsets[k][i] << ' ';
				
				std::cerr << std::endl;
			}
			std::cerr << "dW:" << std::endl;
			for (int k = 0; k < dW.size(); ++k) {				
				for (int i = 0; i < dW[k].size(); ++i) {
					for (int j = 0; j < dW[k][i].size(); ++j)
						std::cerr << dW[k][i][j] << ' ';
					std::cerr << std::endl;
				}
				std::cerr << std::endl;
			}
			std::cerr << "W:" << std::endl;
			for (int k = 0; k < W.size(); ++k) {				
				for (int i = 0; i < W[k].size(); ++i) {
					for (int j = 0; j < W[k][i].size(); ++j)
						std::cerr << W[k][i][j] << ' ';
					std::cerr << std::endl;
				}
				std::cerr << std::endl;
			}
			*/
				
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
					layers[k + 1][j] = enable_offsets ? offsets[k][j] : 0;
					
					for (int i = 0; i < dimensions[k]; ++i)
						layers[k + 1][j] += layers[k][i] * W[k][i][j];
					
					// Normalize
					layers[k + 1][j] = activators[k]->process(layers[k + 1][j]);
				}
			}
			
			
			return layers.back();
		};
		
		void serialize(std::ostream& os) {
			// Format:
			// 1. number of layers
			// 2. size of first layer
			// n+1. size of nth layer
			// n+2. 1 layer activator id
			// 2n+1. n activator id
			// 2n+2. one by one bias matrices
			// 2n+3. offset matrix
			// 2n+4. enable offsets
			os << dimensions.size();
			os << std::endl;
			os << std::endl;
			
			for (int k = 0; k < dimensions.size(); ++k)
				os << dimensions[k] << ' ';
			os << std::endl;
			os << std::endl;
			
			for (int i = 0; i < activators.size(); ++i)
				os << (int) activators[i]->getType() << ' ';
			os << std::endl;
			os << std::endl;
			
			for (int k = 0; k < dimensions.size() - 1; ++k) {
				for (int i = 0; i < dimensions[k]; ++i)
					for (int j = 0; j < dimensions[k + 1]; ++j) 
						os << W[k][i][j] << ' ';
				os << std::endl;
			}
			os << std::endl;
			
			for (int i = 0; i < dimensions.size() - 1; ++i) {
				for (int j = 0; j < dimensions[i + 1]; ++j)
					os << offsets[i][j] << ' ';
				os << std::endl;
			}
			os << std::endl;
			
			os << enable_offsets;
		};
		
		bool deserialize(std::istream& is) {
			int size;
			is >> size;
			dimensions.resize(size);
			
			for (int k = 0; k < dimensions.size(); ++k)
				is >> dimensions[k];
			
			set(dimensions);
			
			for (int i = 0; i < size - 1; ++i) {
				int ac;
				is >> ac;
				
				activators[i] = getActivatorByType((ActivatorType) ac);
			}
			
			for (int k = 0; k < dimensions.size() - 1; ++k)
				for (int i = 0; i < dimensions[k]; ++i)
					for (int j = 0; j < dimensions[k + 1]; ++j) 
						is >> W[k][i][j];
			
			for (int i = 0; i < dimensions.size() - 1; ++i)
				for (int j = 0; j < dimensions[i + 1]; ++j)
					is >> offsets[i][j];
				
			is >> enable_offsets;
			
			return 1;
		};
	};
};