#pragma once

#include "Network.h"

#include <cstdlib>

namespace NNSpace {
	
	// https://habr.com/ru/post/198268/
	
	class SLNet : public Network {
		
	public:
		
		// Weights
		std::vector<std::vector<double>> W01; // Input -> middle layer connections.
		std::vector<std::vector<double>> W12; // Middle -> output layer connections.
		// Offsets
		std::vector<double> middle_offset;
		std::vector<double> output_offset;
		
		bool enable_offsets = 0;
		
		struct {
			int input  = 0;
			int middle = 0;
			int output = 0;
		} dimensions;
		
		NetworkFunction* middle_act, *output_act;
		
		SLNet() : Network() {
			middle_act = new Linear();
			output_act = new Linear();
		};
		
		SLNet(int input, int middle, int output) : Network() {
			set(input, middle, output);
		};
		
		~SLNet() {
			delete middle_act;
			delete output_act;
		};
		
		void set(int input, int middle, int output) {
			W01.clear();
			W12.clear();
			
			W01.resize(input, std::vector<double>(middle));
			W12.resize(middle, std::vector<double>(output));
			
			middle_offset.clear();
			output_offset.clear();
			
			middle_offset.resize(middle);
			output_offset.resize(output);
			
			dimensions.input  = input;
			dimensions.middle = middle;
			dimensions.output = output;
		};
				
		void setLayerActivator(int layer, NetworkFunction* function) {
			if (layer == 1) {
				delete middle_act;
				middle_act = function;
			} else if (layer == 2) {
				delete output_act;
				output_act = function;
			} else
				throw std::runtime_error("Invalid layer number");
		};
		
		virtual void setActivator(NetworkFunction* function) {
			delete middle_act;
			delete output_act;
			
			middle_act = function;
			output_act = function->clone();
		};
		
		void randomize(double dispersion) {
			double scale2 = dispersion * 0.5;
			double v1_MAX = dispersion / RAND_MAX;
			
			for (int i = 0; i < dimensions.input; ++i)
				for (int j = 0; j < dimensions.middle; ++j)
					W01[i][j] = rand() * v1_MAX - scale2;

			for (int i = 0; i < dimensions.middle; ++i)
				for (int j = 0; j < dimensions.output; ++j)
					W12[i][j] = rand() * v1_MAX - scale2;		

			for (int i = 0; i < dimensions.middle; ++i)
				middle_offset[i] = rand() * v1_MAX - scale2;
			
			for (int i = 0; i < dimensions.output; ++i)
				output_offset[i] = rand() * v1_MAX - scale2;
		};
		
		inline void setEnableOffsets(bool e) {
			enable_offsets = e;
		};
		
		// Assume input size match input layer size
		std::vector<double> run(const std::vector<double>& input) {
			std::vector<double> middle(dimensions.middle);
			std::vector<double> output(dimensions.output);
			
			// input -> middle
			for (int j = 0; j < dimensions.middle; ++j) {
				middle[j] = enable_offsets ? middle_offset[j] : enable_offsets;
				
				for (int i = 0; i < dimensions.input; ++i)
					middle[j] += input[i] * W01[i][j];
				
				// After summary, activate
				middle[j] = middle_act->process(middle[j]);
			}
			
			// middle -> output
			for (int j = 0; j < dimensions.output; ++j) {
				output[j] = enable_offsets ? output_offset[j] : 0.0;
				
				for (int i = 0; i < dimensions.middle; ++i)
					output[j] += middle[i] * W12[i][j];
				
				output[j] = output_act->process(output[j]);
			}
				
			return output;
		};
		
		void serialize(std::ostream& os) {
			// Format:
			// 1. number of layers
			// 2. size of input layer
			// 3. size of middle layer
			// 4. size of output layer
			// 5. middle activator id
			// 6. output activator id
			// 7. one by one bias matrices
			// 8. one by one offsets
			// 9. enable offsets
			os << 3;
			os << std::endl;
			os << std::endl;
			os << dimensions.input;
			os << ' ';
			os << dimensions.middle;
			os << ' ';
			os << dimensions.output;
			os << std::endl;
			os << std::endl;
			
			os << (int) middle_act->getType();
			os << ' ';
			os << (int) output_act->getType();
			os << std::endl;
			os << std::endl;
			
			for (int i = 0; i < dimensions.input; ++i)
				for (int j = 0; j < dimensions.middle; ++j) 
					os << W01[i][j] << ' ';
			os << std::endl;
			os << std::endl;
			
			for (int i = 0; i < dimensions.middle; ++i)
				for (int j = 0; j < dimensions.output; ++j) 
					os << W12[i][j] << ' ';
			os << std::endl;
			
			for (int i = 0; i < dimensions.middle; ++i)
				os << middle_offset[i] << ' ';
			os << std::endl;
			
			for (int i = 0; i < dimensions.output; ++i)
				os << output_offset[i] << ' ';
			os << std::endl;
			os << std::endl;
			
			os << enable_offsets;
		};
		
		bool deserialize(std::istream& is) {
			int size;
			is >> size;
			if (size != 3)
				return 0;
			
			is >> dimensions.input;
			is >> dimensions.middle;
			is >> dimensions.output;
			
			int mact;
			int oact;
			
			is >> mact;
			is >> oact;
			
			middle_act = getActivatorByType((ActivatorType) mact);
			output_act = getActivatorByType((ActivatorType) oact);
			
			set(dimensions.input, dimensions.middle, dimensions.output);
			
			for (int i = 0; i < dimensions.input; ++i)
				for (int j = 0; j < dimensions.middle; ++j) 
					is >> W01[i][j];
			for (int i = 0; i < dimensions.middle; ++i)
				for (int j = 0; j < dimensions.output; ++j) 
					is >> W12[i][j];
			
			for (int i = 0; i < dimensions.middle; ++i)
				is >> middle_offset[i];
			
			for (int i = 0; i < dimensions.output; ++i)
				is >> output_offset[i];
			
			is >> enable_offsets;
			
			return 1;
		};
	};
};