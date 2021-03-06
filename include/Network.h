/*
 * (c) Copyright bitrate16 (GPLv3.0) 2020
 */

#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cmath>

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
		virtual NetworkFunction* clone() { return nullptr; };
		ActivatorType getType() { return type; };
	};

	class Linear : public NetworkFunction {

	public:
	
		Linear() { type = ActivatorType::LINEAR; };
		
		double process(double t) { return t; };
		double derivative(double t) { return 1.0; };
		virtual NetworkFunction* clone() { return new Linear(); };
	};

	class Sigmoid : public NetworkFunction {

	public:
	
		Sigmoid() { type = ActivatorType::SIGMOID; };
		
		double process(double t) { return 1 / (1 + std::exp(-t)); };
		double derivative(double t) { return this->process(t) * (1 - this->process(t)); };
		virtual NetworkFunction* clone() { return new Sigmoid; };
	};

	class BipolarSigmoid : public NetworkFunction {

	public:
	
		BipolarSigmoid() { type = ActivatorType::BIPOLAR_SIGMOID; };
		
		double process(double t) { return  2 / (1 + std::exp(-t)) - 1; };
		double derivative(double t) { return 0.5 * (1 + this->process(t)) * (1 - this->process(t)); };
		virtual NetworkFunction* clone() { return new BipolarSigmoid(); };
	};

	class ReLU : public NetworkFunction {

	public:
	
		ReLU() { type = ActivatorType::RELU; };
		
		double process(double t) { return t <= 0.0 ? 0.0 : t; };
		double derivative(double t) { return t <= 0.0 ? 0.0 : 1.0; };
		virtual NetworkFunction* clone() { return new ReLU(); };
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
		virtual NetworkFunction* clone() { return new TanH(); };
	};
	
	inline static NetworkFunction* getActivatorByType(ActivatorType type) {
		switch (type) {
			case ActivatorType::LINEAR:          return new Linear();
			case ActivatorType::SIGMOID:         return new Sigmoid();
			case ActivatorType::BIPOLAR_SIGMOID: return new BipolarSigmoid();
			case ActivatorType::RELU:            return new ReLU();
			case ActivatorType::TANH:            return new TanH();
			default: return new Linear();
		}
	};
	
	inline static NetworkFunction* getActivatorByName(const std::string& name) {
		if (name == "Linear")          return new NNSpace::Linear();
		if (name == "Sigmoid")         return new NNSpace::Sigmoid();
		if (name == "BipolarSigmoid")  return new NNSpace::BipolarSigmoid();
		if (name == "ReLU")            return new NNSpace::ReLU();
		if (name == "TanH")            return new NNSpace::TanH();
		return new Linear();
	};
	
	class Network {
		
	public:
		
		Network() {};
		
		~Network() {};
		
		// Set activation function for passed layers
		virtual void setLayerActivator(int layer, NetworkFunction* function) {};
		
		// Set activation function for all layers
		virtual void setActivator(NetworkFunction* function) {};
		
		// Randomize biases
		virtual void initialize(double dispersion) {}
		
		// Run input data for the output
		virtual std::vector<double> run(const std::vector<double>& input) {};
		
		// Run input data for the output
		virtual void run(const std::vector<double>& input, std::vector<double>& output) {};
		
		// Output network to os as restorable representation form
		virtual void serialize(std::ostream& os) {};
		
		// Read from input & restore state
		// Returns 0 on failture
		virtual bool deserialize(std::istream& is) { return 0; };
	};
};