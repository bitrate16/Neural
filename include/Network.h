#pragma once

#include <vector>
#include <cmath>
#include <iostream>

namespace NNSpace {
	
	class NetworkFunction {

	public:
		
		virtual double process(double t) { return 0; };
		virtual double derivative(double t) { return 0; };
	};

	class Linear : public NetworkFunction {

	public:
		
		double process(double t) { return t; };
		double derivative(double t) { return 1.0; };
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

	class ReLU : public NetworkFunction {

	public:
		
		double process(double t) { return t < 0.0 ? 0.0 : t; };
		double derivative(double t) { return t < 0.0 ? 0.0 : 1.0; };
	};

	class TanH : public NetworkFunction {

	public:
		
		double process(double t) {
			return std::tanh(t); 
		};
		
		double derivative(double t) { 
			double sh = 1.0 / std::cosh(t);   // sech(x) == 1/cosh(x)
			return sh * sh;                   // sech^2(x)
		};
	};

	class Network {
		
	protected:
		
		NetworkFunction *activator;
		
	public:
		
		Network() { activator = new Linear(); };
		
		~Network() { delete activator; };
		
		// Set activation function
		void setFunction(NetworkFunction* function) {
			delete activator;
			activator = function;
		};
		
		// Randomize biases
		virtual void randomize() {};
		
		// Teach on input data
		virtual void train(const std::vector<double>& input, const std::vector<double>& output_teach, double rate) {};
		
		// Teach on input data, returns average error value for output layer
		virtual double train_error(const std::vector<double>& input, const std::vector<double>& output_teach, double rate) { 
			train(input, output_teach, rate); 
			return 0.0;
		};
		
		// Run input data for the output
		virtual std::vector<double> run(const std::vector<double>& input) {};
		
		// Output network to os as restorable representation form
		virtual void serialize(std::ostream& os) {};
		
		// Read from input & restore state
		// Returns 0 on failture
		virtual bool deserialize(std::istream& is) { return 0; };
	};
};