#pragma once

#include <vector>
#include <cmath>
#include <iostream>

namespace NNSpace {
	
	class NetworkFunction {

	public:
		
		virtual double process(double t);
		virtual double derivative(double t);
	};

	class Linear : public NetworkFunction {

	public:
		
		double process(double t) { return t; };
		double derivative(double t) { return 0; };
	};

	class Sigmoid : public NetworkFunction {

	public:
		
		double process(double t) { return 1 / (1 + std::exp(-t)); };
		double derivative(double t) { return this->process(t) * (1 - this->process(t)); };
	};

	class BipolarSigmoid : public NetworkFunction {

	public:
		
		double process(double t) { return  2 / (1 + std::exp(-t)) - 1; };
		double derivative(double t) { return 0.5 * (1 + this->process(t)) * (1 - this->process(t)); };
	};

	class Network {
		
	protected:
		
		NetworkFunction activator = Linear();
		
	public:
		
		Network();
		
		// Set activation function
		void setFunction(NetworkFunction function) {
			activator = function;
		};
		
		// Randomize biases
		virtual void randomize();
		
		// Teach on input data
		virtual void teach(const std::vector<double>& input, const std::vector<double>& output_teach, double rate);
		
		// Run input data for the output
		virtual std::vector<double> run(const std::vector<double>& input);
		
		// Output network to os as restorable representation form
		virtual void serialize(std::ostream& os);
		
		// Read from input & restore state
		// Returns 0 on failture
		virtual bool deserialize(std::istream& is);
	};
};