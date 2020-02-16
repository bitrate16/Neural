/*
 * (c) Copyright bitrate16 (GPLv3.0) 2020
 */

#pragma once

#include <cstdlib>

#include "../MultiLayerNetwork.h"
#include "../SingleLayerNetwork.h"

// Source code for performing training of the networks 
//  using the Backpropagation algorithm.
namespace NNSpace {
	namespace backpropagation {
		

		// M U L T I L A Y E R
		
		// Train using backpropagation
		// Assume input, output_teach size match input, output layer size
		// net          - input network to train
		// input        - input data to train on
		// output_teach - desired output result
		// rate         - teach rate value
		void train(NNSpace::MLNet& net, const std::vector<double>& input, const std::vector<double>& output_teach, double rate) {
			std::vector<std::vector<double>> layers(net.dimensions.size()); // [0-N]
			layers[0] = input;
			
			std::vector<std::vector<double>> layers_raw(net.dimensions.size() - 1); // [1-N]
			
			// Regular process
			for (int k = 0; k < net.dimensions.size() - 1; ++k) {
				layers[k + 1].resize(net.dimensions[k + 1]);
				layers_raw[k].resize(net.dimensions[k + 1]);
				
				// calculate RAW layer outputs & normalize them
				for (int j = 0; j < net.dimensions[k + 1]; ++j) {
					layers_raw[k][j] = net.enable_offsets ? net.offsets[k][j] : 0;
					
					for (int i = 0; i < net.dimensions[k]; ++i)
						layers_raw[k][j] += layers[k][i] * net.W[k][i][j];
					
					// Normalize
					layers[k + 1][j] = net.activators[k]->process(layers_raw[k][j]);
				}
			}
			
			// Weights correction
			std::vector<std::vector<std::vector<double>>> dW(net.dimensions.size() - 1);
			
			for (int i = 0; i < net.dimensions.size() - 1; ++i) {
				dW[i].resize(net.dimensions[i]);
				for (int j = 0; j < net.dimensions[i]; ++j)
					dW[i][j].resize(net.dimensions[i + 1]);
			}
			
			// Offsets correction
			std::vector<std::vector<double>> doffset(net.dimensions.size() - 1);
			// Sigmas
			std::vector<std::vector<double>> sigma(net.dimensions.size() - 1);
			
			for (int i = 0; i < net.dimensions.size() - 1; ++i) {
				doffset[i].resize(net.dimensions[i + 1]);
				sigma[i].resize(net.dimensions[i + 1]);
			}
			
			// Calculate sigmas
			for (int i = 0; i < net.dimensions.back(); ++i) { // K-2, K-1
				double dv = output_teach[i] - layers.back()[i];
				sigma.back()[i] = dv * net.activators.back()->derivative(layers_raw.back()[i]);
			}
			
			for (int k = (net.dimensions.size() - 1) - 2; k >= 0; --k) // K-3, K-2,, ..
				for (int i = 0; i < net.dimensions[k + 1]; ++i) {
					for (int j = 0; j < net.dimensions[k + 2]; ++j)
						sigma[k][i] += sigma[k + 1][j] * net.W[k + 1][i][j];
					
					sigma[k][i] *= net.activators[k + 0]->derivative(layers_raw[k + 1 - 1][i]); // layers_raw[k + 1]
				}
			
			// Calculate weights correction
			for (int k = 0; k < net.dimensions.size() - 1; ++k)
				for (int i = 0; i < net.dimensions[k]; ++i)
					for (int j = 0; j < net.dimensions[k + 1]; ++j)
						dW[k][i][j] = rate * sigma[k][j] * layers[k][i];
					
			// Calculate offset correction
			for (int k = 0; k < net.dimensions.size() - 1; ++k)
				for (int i = 0; i < net.dimensions[k + 1]; ++i)
					doffset[k][i] = rate * sigma[k][i];
					
			// Balance weights
			for (int k = 0; k < net.dimensions.size() - 1; ++k)
				for (int i = 0; i < dW[k].size(); ++i)
					for (int j = 0; j < dW[k][i].size(); ++j)
						net.W[k][i][j] += dW[k][i][j];
					
			// Balance offsets
			for (int k = 0; k < net.dimensions.size() - 1; ++k)
				for (int i = 0; i < net.dimensions[k + 1]; ++i)
					net.offsets[k][i] += doffset[k][i];
		};
		
		// Perform training of the network and calculating error value on the output layer
		// net          - input network to train
		// input        - input data to train on
		// output_teach - desired output result
		// rate         - teach rate value
		// Ltype        - type of error calculation:
		//  1 - L1
		//  2 - L2
		double train_error(NNSpace::MLNet& net, int Ltype, const std::vector<double>& input, const std::vector<double>& output_teach, double rate) {
			long double out_error_value = 0.0;
			std::vector<std::vector<double>> layers(net.dimensions.size()); // [0-N]
			layers[0] = input;
			
			std::vector<std::vector<double>> layers_raw(net.dimensions.size() - 1); // [1-N]
			
			// Regular process
			for (int k = 0; k < net.dimensions.size() - 1; ++k) {
				layers[k + 1].resize(net.dimensions[k + 1]);
				layers_raw[k].resize(net.dimensions[k + 1]);
				
				// calculate RAW layer outputs & normalize them
				for (int j = 0; j < net.dimensions[k + 1]; ++j) {
					layers_raw[k][j] = net.enable_offsets ? net.offsets[k][j] : 0;
					
					for (int i = 0; i < net.dimensions[k]; ++i)
						layers_raw[k][j] += layers[k][i] * net.W[k][i][j];
					
					// Normalize
					layers[k + 1][j] = net.activators[k]->process(layers_raw[k][j]);
				}
			}
			
			// Weights correction
			std::vector<std::vector<std::vector<double>>> dW(net.dimensions.size() - 1);
			
			for (int i = 0; i < net.dimensions.size() - 1; ++i) {
				dW[i].resize(net.dimensions[i]);
				for (int j = 0; j < net.dimensions[i]; ++j)
					dW[i][j].resize(net.dimensions[i + 1]);
			}
			
			// Offsets correction
			std::vector<std::vector<double>> doffset(net.dimensions.size() - 1);
			// Sigmas
			std::vector<std::vector<double>> sigma(net.dimensions.size() - 1);
			
			for (int i = 0; i < net.dimensions.size() - 1; ++i) {
				doffset[i].resize(net.dimensions[i + 1]);
				sigma[i].resize(net.dimensions[i + 1]);
			}
			
			// Calculate sigmas
			for (int i = 0; i < net.dimensions.back(); ++i) { // K-2, K-1
				double dv = output_teach[i] - layers.back()[i];
				sigma.back()[i] = dv * net.activators.back()->derivative(layers_raw.back()[i]);
				
				if (Ltype == 2)
					out_error_value += dv * dv;
				else if (Ltype == 1)
					out_error_value += std::fabs(dv);
			}
			
			for (int k = (net.dimensions.size() - 1) - 2; k >= 0; --k) // K-3, K-2,, ..
				for (int i = 0; i < net.dimensions[k + 1]; ++i) {
					for (int j = 0; j < net.dimensions[k + 2]; ++j)
						sigma[k][i] += sigma[k + 1][j] * net.W[k + 1][i][j];
					
					sigma[k][i] *= net.activators[k + 0]->derivative(layers_raw[k + 1 - 1][i]); // layers_raw[k + 1]
				}
			
			// Calculate weights correction
			for (int k = 0; k < net.dimensions.size() - 1; ++k)
				for (int i = 0; i < net.dimensions[k]; ++i)
					for (int j = 0; j < net.dimensions[k + 1]; ++j)
						dW[k][i][j] = rate * sigma[k][j] * layers[k][i];
					
			// Calculate offset correction
			for (int k = 0; k < net.dimensions.size() - 1; ++k)
				for (int i = 0; i < net.dimensions[k + 1]; ++i)
					doffset[k][i] = rate * sigma[k][i];
					
			// Balance weights
			for (int k = 0; k < net.dimensions.size() - 1; ++k)
				for (int i = 0; i < dW[k].size(); ++i)
					for (int j = 0; j < dW[k][i].size(); ++j)
						net.W[k][i][j] += dW[k][i][j];
					
			// Balance offsets
			for (int k = 0; k < net.dimensions.size() - 1; ++k)
				for (int i = 0; i < net.dimensions[k + 1]; ++i)
					net.offsets[k][i] += doffset[k][i];
				
				
			if (Ltype == 2)
				return std::sqrt(out_error_value / (double) net.dimensions.back());
			else if (Ltype == 1)
				return out_error_value / (double) net.dimensions.back();
			return 0.0;
		};
	
		
		// S I N G L E L A Y E R
		
		// Train using backpropagation
		// Assume input, output_teach size match input, output layer size
		// net          - input network to train
		// input        - input data to train on
		// output_teach - desired output result
		// rate         - teach rate value
		void train(NNSpace::SLNet& net, const std::vector<double>& input, const std::vector<double>& output_teach, double rate) {
			std::vector<double> middle_raw(net.dimensions.middle);
			std::vector<double> middle(net.dimensions.middle);
			std::vector<double> output_raw(net.dimensions.output);
			std::vector<double> output(net.dimensions.output);
			
			// input -> middle
			for (int j = 0; j < net.dimensions.middle; ++j) {
				middle_raw[j] = net.enable_offsets ? net.middle_offset[j] : 0;
				
				for (int i = 0; i < net.dimensions.input; ++i)
					middle_raw[j] += input[i] * net.W01[i][j];
				
				// After summary, activate
				middle[j] = net.middle_act->process(middle_raw[j]);
			}
			
			// middle -> output
			for (int j = 0; j < net.dimensions.output; ++j) {
				output_raw[j] = net.enable_offsets ? net.output_offset[j] : 0;
				
				for (int i = 0; i < net.dimensions.middle; ++i)
					output_raw[j] += middle[i] * net.W12[i][j];
				
				output[j] = net.output_act->process(output_raw[j]);
			}
				
			// Calculate sigma (error value) for output layer
			std::vector<double> sigma_output(net.dimensions.output);
			for (int i = 0; i < net.dimensions.output; ++i) {
				double dv = output_teach[i] - output[i];
				sigma_output[i] = dv * net.output_act->derivative(output_raw[i]);
			}
			
			// Calculate bias correction for middle-output
			std::vector<std::vector<double>> dW12;
			dW12.resize(net.dimensions.middle, std::vector<double>(output));
			for (int i = 0; i < net.dimensions.middle; ++i)
				for (int j = 0; j < net.dimensions.output; ++j)
					dW12[i][j] = rate * sigma_output[j] * middle[i];
				
			// Calculate offset correction for middle-output
			std::vector<double> do_offset(net.dimensions.output);
			
			for (int i = 0; i < net.dimensions.output; ++i)
				do_offset[i] = rate * sigma_output[i];
				
			// Calculate sigma (error value) for middle layer
			std::vector<double> sigma_middle_raw(net.dimensions.middle);
			for (int i = 0; i < net.dimensions.middle; ++i)
				for (int j = 0; j < net.dimensions.output; ++j)
					sigma_middle_raw[i] += sigma_output[j] * net.W12[i][j];
			
			// Multiply by activator derivative value
			std::vector<double> sigma_middle(net.dimensions.middle);
			
			// Calculate bias balance
			std::vector<std::vector<double>> dW01;
			dW01.resize(net.dimensions.input, std::vector<double>(middle));
			
			for (int j = 0; j < net.dimensions.middle; ++j) {
				// Multiply by activator derivative
				sigma_middle[j] = sigma_middle_raw[j] * net.middle_act->derivative(middle_raw[j]);
				
				for (int i = 0; i < net.dimensions.input; ++i)
					dW01[i][j] = rate * sigma_middle[j] * input[i];
			}
				
			// Calculate offset correction for middle-output
			std::vector<double> dm_offset(net.dimensions.middle);
			
			for (int i = 0; i < net.dimensions.middle; ++i)
				dm_offset[i] = rate * sigma_middle[i];
			
			// Balance weights
			for (int i = 0; i < net.dimensions.input; ++i)
				for (int j = 0; j < net.dimensions.middle; ++j)
					net.W01[i][j] += dW01[i][j];

			for (int i = 0; i < net.dimensions.middle; ++i)
				for (int j = 0; j < net.dimensions.output; ++j)
					net.W12[i][j] += dW12[i][j];	
				
			// Balance offsets
			for (int i = 0; i < net.dimensions.middle; ++i)
				net.middle_offset[i] += dm_offset[i];
			
			for (int i = 0; i < net.dimensions.output; ++i)
				net.output_offset[i] += do_offset[i];
		};
	
		// Perform training of the network and calculating error value on the output layer
		// net          - input network to train
		// input        - input data to train on
		// output_teach - desired output result
		// rate         - teach rate value
		// Ltype        - type of error calculation:
		//  1 - L1
		//  2 - L2
		double train_error(NNSpace::SLNet& net, int Ltype, const std::vector<double>& input, const std::vector<double>& output_teach, double rate) {
			long double out_error_value = 0.0;
			std::vector<double> middle_raw(net.dimensions.middle);
			std::vector<double> middle(net.dimensions.middle);
			std::vector<double> output_raw(net.dimensions.output);
			std::vector<double> output(net.dimensions.output);
			
			// input -> middle
			for (int j = 0; j < net.dimensions.middle; ++j) {
				middle_raw[j] = net.enable_offsets ? net.middle_offset[j] : 0;
				
				for (int i = 0; i < net.dimensions.input; ++i)
					middle_raw[j] += input[i] * net.W01[i][j];
				
				// After summary, activate
				middle[j] = net.middle_act->process(middle_raw[j]);
			}
			
			// middle -> output
			for (int j = 0; j < net.dimensions.output; ++j) {
				output_raw[j] = net.enable_offsets ? net.output_offset[j] : 0;
				
				for (int i = 0; i < net.dimensions.middle; ++i)
					output_raw[j] += middle[i] * net.W12[i][j];
				
				output[j] = net.output_act->process(output_raw[j]);
			}
				
			// Calculate sigma (error value) for output layer
			std::vector<double> sigma_output(net.dimensions.output);
			for (int i = 0; i < net.dimensions.output; ++i) {
				double dv = output_teach[i] - output[i];
				sigma_output[i] = dv * net.output_act->derivative(output_raw[i]);
				
				// Calculate total error
				if (Ltype == 2)
					out_error_value += dv * dv;
				else if (Ltype == 1)
					out_error_value += std::fabs(dv);
			}
			
			// Calculate bias correction for middle-output
			std::vector<std::vector<double>> dW12;
			dW12.resize(net.dimensions.middle, std::vector<double>(output));
			for (int i = 0; i < net.dimensions.middle; ++i)
				for (int j = 0; j < net.dimensions.output; ++j)
					dW12[i][j] = rate * sigma_output[j] * middle[i];
				
			// Calculate offset correction for middle-output
			std::vector<double> do_offset(net.dimensions.output);
			
			for (int i = 0; i < net.dimensions.output; ++i)
				do_offset[i] = rate * sigma_output[i];
				
			// Calculate sigma (error value) for middle layer
			std::vector<double> sigma_middle_raw(net.dimensions.middle);
			for (int i = 0; i < net.dimensions.middle; ++i)
				for (int j = 0; j < net.dimensions.output; ++j)
					sigma_middle_raw[i] += sigma_output[j] * net.W12[i][j];
			
			// Multiply by activator derivative value
			std::vector<double> sigma_middle(net.dimensions.middle);
			
			// Calculate bias balance
			std::vector<std::vector<double>> dW01;
			dW01.resize(net.dimensions.input, std::vector<double>(middle));
			
			for (int j = 0; j < net.dimensions.middle; ++j) {
				// Multiply by activator derivative
				sigma_middle[j] = sigma_middle_raw[j] * net.middle_act->derivative(middle_raw[j]);
				
				for (int i = 0; i < net.dimensions.input; ++i)
					dW01[i][j] = rate * sigma_middle[j] * input[i];
			}
				
			// Calculate offset correction for middle-output
			std::vector<double> dm_offset(net.dimensions.middle);
			
			for (int i = 0; i < net.dimensions.middle; ++i)
				dm_offset[i] = rate * sigma_middle[i];
			
			// Balance weights
			for (int i = 0; i < net.dimensions.input; ++i)
				for (int j = 0; j < net.dimensions.middle; ++j)
					net.W01[i][j] += dW01[i][j];

			for (int i = 0; i < net.dimensions.middle; ++i)
				for (int j = 0; j < net.dimensions.output; ++j)
					net.W12[i][j] += dW12[i][j];	
				
			// Balance offsets
			for (int i = 0; i < net.dimensions.middle; ++i)
				net.middle_offset[i] += dm_offset[i];
			
			for (int i = 0; i < net.dimensions.output; ++i)
				net.output_offset[i] += do_offset[i];
			
			
			if (Ltype == 2)
				return std::sqrt(out_error_value / (double) net.dimensions.output);
			else if (Ltype == 1)
				return out_error_value / (double) net.dimensions.output;
			return 0.0;
		};
	};
};