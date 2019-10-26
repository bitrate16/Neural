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
		
		struct {
			int input  = 0;
			int middle = 0;
			int output = 0;
		} dimensions;
		
		SLNetwork() : Network() {};
		
		SLNetwork(int input, int middle, int output) : Network() {
			set(input, middle, output);
		};
		
		void set(int input, int middle, int output) {
			W01.resize(input, std::vector<double>(middle));
			W12.resize(middle, std::vector<double>(output));
			middle_offset.resize(middle);
			output_offset.resize(output);
			
			dimensions.input  = input;
			dimensions.middle = middle;
			dimensions.output = output;
		};
		
		void randomize() {
			double v1_MAX = 2.0 / RAND_MAX;
			
			for (int i = 0; i < dimensions.input; ++i)
				for (int j = 0; j < dimensions.middle; ++j)
					W01[i][j] = rand() * v1_MAX - 1.0;

			for (int i = 0; i < dimensions.middle; ++i)
				for (int j = 0; j < dimensions.output; ++j)
					W12[i][j] = rand() * v1_MAX - 1.0;		

			double v2_MAX = 1.0 / RAND_MAX;
			
			for (int i = 0; i < dimensions.middle; ++i)
				middle_offset[i] = rand() * v2_MAX;
			
			for (int i = 0; i < dimensions.output; ++i)
				output_offset[i] = rand() * v2_MAX;
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
				middle_raw[j] = middle_offset[j];
				
				for (int i = 0; i < dimensions.input; ++i)
					middle_raw[j] += input[i] * W01[i][j];
				
				// After summary, activate
				middle[j] = activator->process(middle_raw[j]);
			}
			
			// middle -> output
			for (int j = 0; j < dimensions.output; ++j) {
				output_raw[j] = output_offset[j];
				
				for (int i = 0; i < dimensions.middle; ++i)
					output_raw[j] += middle[i] * W12[i][j];
				
				output[j] = activator->process(output_raw[j]);
			}
				
			// Calculate sigma (error value) for output layer
			std::vector<double> sigma_output(dimensions.output);
			for (int i = 0; i < dimensions.output; ++i)
				sigma_output[i] = (output_teach[i] - output[i]) * activator->derivative(output_raw[i]);
			
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
				sigma_middle[j] = sigma_middle_raw[j] * activator->derivative(middle_raw[j]);
				
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
				middle[j] = activator->process(middle_raw[j]);
			}
			
			// middle -> output
			for (int j = 0; j < dimensions.output; ++j) {
				output_raw[j] = output_offset[j];
				
				for (int i = 0; i < dimensions.middle; ++i)
					output_raw[j] += middle[i] * W12[i][j];
				
				output[j] = activator->process(output_raw[j]);
			}
				
			// Calculate sigma (error value) for output layer
			std::vector<double> sigma_output(dimensions.output);
			for (int i = 0; i < dimensions.output; ++i) {
				double dv = output_teach[i] - output[i];
				sigma_output[i] = dv * activator->derivative(output_raw[i]);
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
				sigma_middle[j] = sigma_middle_raw[j] * activator->derivative(middle_raw[j]);
				
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
				middle[j] = middle_offset[j];
				
				for (int i = 0; i < dimensions.input; ++i)
					middle[j] += input[i] * W01[i][j];
				
				// After summary, activate
				middle[j] = activator->process(middle[j]);
			}
			
			// middle -> output
			for (int j = 0; j < dimensions.output; ++j) {
				output[j] = output_offset[j];
				
				for (int i = 0; i < dimensions.middle; ++i)
					output[j] += middle[i] * W12[i][j];
				
				output[j] = activator->process(output[j]);
			}
				
			return output;
		};
		
		void serialize(std::ostream& os) {
			// Format:
			// 1. number of layers
			// 2. size of first layer
			// n+1. size of nth layer
			// n+2. one by one bias matrices
			os << 3;
			os << ' ';
			os << dimensions.input;
			os << ' ';
			os << dimensions.middle;
			os << ' ';
			os << dimensions.output;
			os << ' ';
			for (int i = 0; i < dimensions.input; ++i)
				for (int j = 0; j < dimensions.middle; ++j) 
					os << W01[i][j] << ' ';
			for (int i = 0; i < dimensions.middle; ++i)
				for (int j = 0; j < dimensions.output; ++j) 
					os << W12[i][j] << ' ';
		};
		
		bool deserialize(std::istream& is) {
			int size;
			is >> size;
			if (size != 3)
				return 0;
			
			is >> dimensions.input;
			is >> dimensions.middle;
			is >> dimensions.output;
			for (int i = 0; i < dimensions.input; ++i)
				for (int j = 0; j < dimensions.middle; ++j) 
					is >> W01[i][j];
			for (int i = 0; i < dimensions.middle; ++i)
				for (int j = 0; j < dimensions.output; ++j) 
					is >> W12[i][j];
			return 1;
		};
	};
};