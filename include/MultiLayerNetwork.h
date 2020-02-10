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
		
		void randomize(double dispersion) {
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