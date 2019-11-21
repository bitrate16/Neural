#pragma once

#include <vector>
#include <cmath>
#include <iostream>

namespace NNSpace {

	enum ActivatorType {
		LINEAR, 
		SIGMOID, 
		BIPOLAR_SIGMOID, 
		RELU, 
		TANH
	};
	
	class NetworkFunction {

	protected:
	
		ActivatorType type;

	public:
		
		virtual double process(double t) { return 0; };
		virtual double derivative(double t) { return 0; };
		ActivatorType getType() { return type; };
	};

	class Linear : public NetworkFunction {

	public:
	
		Linear() { type = ActivatorType::LINEAR; };
		
		double process(double t) { return t; };
		double derivative(double t) { return 1.0; };
	};

	class Sigmoid : public NetworkFunction {

	public:
	
		Sigmoid() { type = ActivatorType::SIGMOID; };
		
		double process(double t) { return 1 / (1 + std::exp(-t)); };
		double derivative(double t) { return this->process(t) * (1 - this->process(t)); };
	};

	class BipolarSigmoid : public NetworkFunction {

	public:
	
		BipolarSigmoid() { type = ActivatorType::BIPOLAR_SIGMOID; };
		
		double process(double t) { return  2 / (1 + std::exp(-t)) - 1; };
		double derivative(double t) { return 0.5 * (1 + this->process(t)) * (1 - this->process(t)); };
	};

	class ReLU : public NetworkFunction {

	public:
	
		ReLU() { type = ActivatorType::RELU; };
		
		double process(double t) { return t <= 0.0 ? 0.0 : t; };
		double derivative(double t) { return t <= 0.0 ? 0.0 : 1.0; };
	};

	class TanH : public NetworkFunction {

	public:
	
		TanH() { type = ActivatorType::TANH; };
		
		double process(double t) {
			return std::tanh(t); 
		};
		
		double derivative(double t) { 
			double sh = 1.0 / std::cosh(t);   // sech(x) == 1/cosh(x)
			return sh * sh;                   // sech^2(x)
		};
	};
	
	NetworkFunction* getActivatorByType(ActivatorType type) {
		switch (type) {
			case ActivatorType::LINEAR:          return new Linear();
			case ActivatorType::SIGMOID:         return new Sigmoid();
			case ActivatorType::BIPOLAR_SIGMOID: return new BipolarSigmoid();
			case ActivatorType::RELU:            return new ReLU();
			case ActivatorType::TANH:            return new TanH();
			default: throw std::runtime_error("Unsupported activator type");
		}
	};

	class Network {
		
	public:
		
		Network() {};
		
		~Network() {};
		
		// Set activation function for passed layers
		virtual void setLayerActivator(int layer, NetworkFunction* function) {};
		
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