#include <iostream>
#include <string>

#include "MultistartCutoff.h"
#include "math_func.h"
#include "math_func_util.h"

// g++ -O3 src/generate_set.cpp -o bin/generate_set -Iinclude && ./bin/generate_set
// ./bin/generate_set 0.0 1.0 10000 "sin(t * 3.14 * 2.0) * 0.5 + 0.5" set.nse

int main(int argc, char** argv) {
	double start, end;
	int amount;
	std::string funct;
	std::string filename;
	
	// Check args
	if (argc < 6) {
		std::cout << "Usage: generate_set <start> <end> <amount> <function> <filename>" << std::endl;
		return 0;
	}
	
	// Parse args
	start    = std::stod(argv[1]);
	end      = std::stod(argv[2]);
	amount   = std::stoi(argv[3]);
	funct    = argv[4];
	filename = argv[5];
	
	math_func::func* func = math_func::parse(funct);
	if (!func) {
		std::cout << "Error parsing" << std::endl;
		return 0;
	}
	
	math_func::func* opt = math_func::optimize(func);
	delete func;
	
	std::map<std::string, double> values;
	std::map<std::string, std::function<double(const std::vector<double>&)>> functions;
	
	// Init default functions
	functions["sin"] = [](std::vector<double> t) {
		return std::sin(t.size() == 0 ? 0.0 : t[0]);
	};
	functions["cos"] = [](std::vector<double> t) {
		return std::cos(t.size() == 0 ? 0.0 : t[0]);
	};
	functions["tan"] = [](std::vector<double> t) {
		return std::tan(t.size() == 0 ? 0.0 : t[0]);
	};
	functions["ctan"] = [](std::vector<double> t) {
		return std::cos(t.size() == 0 ? 0.0 : t[0]) / std::sin(t.size() == 0 ? 0.0 : t[0]);
	};
	functions["exp"] = [](std::vector<double> t) {
		return std::exp(t.size() == 0 ? 0.0 : t[0]);
	};
	functions["pow"] = [](std::vector<double> t) {
		if (t.size() == 0)
			return 0.0;
		if (t.size() == 1)
			return 1.0;
		return std::pow(t[0], t[1]);
	};
	functions["log"] = [](std::vector<double> t) {
		if (t.size() == 0)
			return 0.0;
		return std::log(t[0]);
	};
	functions["arcsin"] = [](std::vector<double> t) {
		return std::asin(t.size() == 0 ? 0.0 : t[0]);
	};
	functions["arccos"] = [](std::vector<double> t) {
		return std::acos(t.size() == 0 ? 0.0 : t[0]);
	};
	functions["arctan"] = [](std::vector<double> t) {
		return std::atan(t.size() == 0 ? 0.0 : t[0]);
	};
	
	NNSpace::generate_linear_set([&values, &functions, &opt](double t) {
		values["t"] = t;
		return opt->evaluate(values, functions);
	}, start, end, amount, filename, 1, 1);
	
	delete opt;
};