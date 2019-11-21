#pragma once

#include "Network.h"

#include <cstdlib>

namespace NNSpace {
	
	// https://habr.com/ru/post/198268/
	
	class SLNetwork : public Network {
		
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
		
		SLNetwork() : Network() {
			middle_act = new Linear();
			output_act = new Linear();
		};
		
		SLNetwork(int input, int middle, int output) : Network() {
			set(input, middle, output);
		};
		
		~SLNetwork() {
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
		
		void randomize() {
			double v1_MAX = 1.0 / RAND_MAX;
			
			for (int i = 0; i < dimensions.input; ++i)
				for (int j = 0; j < dimensions.middle; ++j)
					W01[i][j] = rand() * v1_MAX - 0.5;

			for (int i = 0; i < dimensions.middle; ++i)
				for (int j = 0; j < dimensions.output; ++j)
					W12[i][j] = rand() * v1_MAX - 0.5;		

			for (int i = 0; i < dimensions.middle; ++i)
				middle_offset[i] = rand() * v1_MAX - 0.5;
			
			for (int i = 0; i < dimensions.output; ++i)
				output_offset[i] = rand() * v1_MAX - 0.5;
		};
		
		inline void setEnableOffsets(bool e) {
			enable_offsets = e;
		};
		
		// Teach using backpropagation
		// Assume input, output_teach size match input, output layer size
		void train(const std::vector<double>& input, const std::vector<double>& output_teach, double rate) {
			std::vector<double> middle_raw(dimensions.middle);
			std::vector<double> middle(dimensions.middle);
			std::vector<double> output_raw(dimensions.output);
			std::vector<double> output(dimensions.output);
			
			// input -> middle
			for (int j = 0; j < dimensions.middle; ++j) {
				middle_raw[j] = enable_offsets ? middle_offset[j] : 0.0;
				
				for (int i = 0; i < dimensions.input; ++i)
					middle_raw[j] += input[i] * W01[i][j];
				
				// After summary, activate
				middle[j] = middle_act->process(middle_raw[j]);
			}
			
			// middle -> output
			for (int j = 0; j < dimensions.output; ++j) {
				output_raw[j] = enable_offsets ? output_offset[j] : 0.0;
				
				for (int i = 0; i < dimensions.middle; ++i)
					output_raw[j] += middle[i] * W12[i][j];
				
				output[j] = output_act->process(output_raw[j]);
			}
				
			// Calculate sigma (error value) for output layer
			std::vector<double> sigma_output(dimensions.output);
			for (int i = 0; i < dimensions.output; ++i)
				sigma_output[i] = (output_teach[i] - output[i]) * output_act->derivative(output_raw[i]);
			
			// Calculate bias correction for middle-output
			std::vector<std::vector<double>> dW12;
			dW12.resize(dimensions.middle, std::vector<double>(output));
			for (int i = 0; i < dimensions.middle; ++i)
				for (int j = 0; j < dimensions.output; ++j)
					dW12[i][j] = rate * sigma_output[j] * middle[i];
				
			// Calculate offset correction for middle-output
			std::vector<double> do_offset;
			do_offset.resize(dimensions.output);
			for (int i = 0; i < dimensions.output; ++i)
				do_offset[i] = rate * sigma_output[i];
				
			// Calculate sigma (error value) for middle layer
			std::vector<double> sigma_middle_raw(dimensions.middle);
			for (int i = 0; i < dimensions.middle; ++i)
				for (int j = 0; j < dimensions.output; ++j)
					sigma_middle_raw[i] += sigma_output[j] * W12[i][j];
			
			// Multiply by activator derivative value
			std::vector<double> sigma_middle(dimensions.middle);
			
			// Calculate bias balance
			std::vector<std::vector<double>> dW01;
			dW01.resize(dimensions.input, std::vector<double>(middle));
			for (int j = 0; j < dimensions.middle; ++j) {
				// Multiply by activator derivative
				sigma_middle[j] = sigma_middle_raw[j] * middle_act->derivative(middle_raw[j]);
				
				for (int i = 0; i < dimensions.input; ++i)
					dW01[i][j] = rate * sigma_middle[j] * input[i];
			}
				
			// Calculate offset correction for middle-output
			std::vector<double> dm_offset;
			dm_offset.resize(dimensions.middle);
			for (int i = 0; i < dimensions.middle; ++i)
				dm_offset[i] = rate * sigma_middle[i];
			
			// Balance weights
			for (int i = 0; i < dimensions.input; ++i)
				for (int j = 0; j < dimensions.middle; ++j)
					W01[i][j] += dW01[i][j];

			for (int i = 0; i < dimensions.middle; ++i)
				for (int j = 0; j < dimensions.output; ++j)
					W12[i][j] += dW12[i][j];	
				
			// Balance offsets
			for (int i = 0; i < dimensions.middle; ++i)
				middle_offset[i] += dm_offset[i];
			
			for (int i = 0; i < dimensions.output; ++i)
				output_offset[i] += do_offset[i];
		};
		
		double train_error(const std::vector<double>& input, const std::vector<double>& output_teach, double rate) {
			std::vector<double> middle_raw(dimensions.middle);
			std::vector<double> middle(dimensions.middle);
			std::vector<double> output_raw(dimensions.output);
			std::vector<double> output(dimensions.output);
			double out_error_value = 0.0;
			
			// input -> middle
			for (int j = 0; j < dimensions.middle; ++j) {
				middle_raw[j] = middle_offset[j];
				
				for (int i = 0; i < dimensions.input; ++i)
					middle_raw[j] += input[i] * W01[i][j];
				
				// After summary, activate
				middle[j] = middle_act->process(middle_raw[j]);
			}
			
			// middle -> output
			for (int j = 0; j < dimensions.output; ++j) {
				output_raw[j] = output_offset[j];
				
				for (int i = 0; i < dimensions.middle; ++i)
					output_raw[j] += middle[i] * W12[i][j];
				
				output[j] = output_act->process(output_raw[j]);
			}
				
			// Calculate sigma (error value) for output layer
			std::vector<double> sigma_output(dimensions.output);
			for (int i = 0; i < dimensions.output; ++i) {
				double dv = output_teach[i] - output[i];
				sigma_output[i] = dv * output_act->derivative(output_raw[i]);
				out_error_value += dv;
			}
			
			// Calculate bias correction for middle-output
			std::vector<std::vector<double>> dW12;
			dW12.resize(dimensions.middle, std::vector<double>(output));
			for (int i = 0; i < dimensions.middle; ++i)
				for (int j = 0; j < dimensions.output; ++j)
					dW12[i][j] = rate * sigma_output[j] * middle[i];
				
			// Calculate offset correction for middle-output
			std::vector<double> do_offset;
			do_offset.resize(dimensions.output);
			for (int i = 0; i < dimensions.output; ++i)
				do_offset[i] = rate * sigma_output[i];
				
			// Calculate sigma (error value) for middle layer
			std::vector<double> sigma_middle_raw(dimensions.middle);
			for (int i = 0; i < dimensions.middle; ++i)
				for (int j = 0; j < dimensions.output; ++j)
					sigma_middle_raw[i] += sigma_output[j] * W12[i][j];
			
			// Multiply by activator derivative value
			std::vector<double> sigma_middle(dimensions.middle);
			
			// Calculate bias balance
			std::vector<std::vector<double>> dW01;
			dW01.resize(dimensions.input, std::vector<double>(middle));
			for (int j = 0; j < dimensions.middle; ++j) {
				// Multiply by activator derivative
				sigma_middle[j] = sigma_middle_raw[j] * middle_act->derivative(middle_raw[j]);
				
				for (int i = 0; i < dimensions.input; ++i)
					dW01[i][j] = rate * sigma_middle[j] * input[i];
			}
				
			// Calculate offset correction for middle-output
			std::vector<double> dm_offset;
			dm_offset.resize(dimensions.middle);
			for (int i = 0; i < dimensions.middle; ++i)
				dm_offset[i] = rate * sigma_middle[i];
			
			// Balance weights
			for (int i = 0; i < dimensions.input; ++i)
				for (int j = 0; j < dimensions.middle; ++j)
					W01[i][j] += dW01[i][j];

			for (int i = 0; i < dimensions.middle; ++i)
				for (int j = 0; j < dimensions.output; ++j)
					W12[i][j] += dW12[i][j];	
				
			// Balance offsets
			for (int i = 0; i < dimensions.middle; ++i)
				middle_offset[i] += dm_offset[i];
			
			for (int i = 0; i < dimensions.output; ++i)
				output_offset[i] += do_offset[i];
			
			return out_error_value / (double) dimensions.output;
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
			os << ' ';
			os << dimensions.input;
			os << ' ';
			os << dimensions.middle;
			os << ' ';
			os << dimensions.output;
			os << std::endl;
			
			os << (int) middle_act->getType();
			os << ' ';
			os << (int) output_act->getType();
			os << std::endl;
			
			for (int i = 0; i < dimensions.input; ++i)
				for (int j = 0; j < dimensions.middle; ++j) 
					os << W01[i][j] << ' ';
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