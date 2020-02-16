#include "math_func_util.h"
#include "NetTestCommon.h"
#include "math_func.h"
#include "pargs.h"

#include <iostream>

/*
 * (c) Copyright bitrate16 (GPLv3.0) 2020
 * Generates testing set for 2D approximation
 * Agruments:
 *  --function="%"   Approximation function formula
 *  --start=%        Approximation interval start
 *  --end=%          Approximation interval end
 *  --count=%        Points count
 *  --output=%       Output filename
 *  --random=%       Use random instead of linear
 *
 * Make:
 * g++ src/train_test/gen_set_2d.cpp -o bin/gen_set_2d -O3 --std=c++17 -Iinclude -lstdc++fs
 *
 * Example:
 * ./bin/gen_set_2d --function="sin(t*3.14*2.0)*0.5+0.5" --count=1000 --output=data/sin_1000.mset --random=true
 */

int main(int argc, const char** argv) {
	pargs::pargs args(argc, argv);
	
	std::string output   = args["--output"]   && args["--output"]->is_string()   ? args["--output"]->string()   : "set.mset";
	double start         = args["--start"]    && args["--start"]->is_real()      ? args["--start"]->real()      : 0.0;
	double end           = args["--end"]      && args["--end"]->is_real()        ? args["--end"]->real()        : 1.0;
	std::string function = args["--function"] && args["--function"]->is_string() ? args["--function"]->string() : "t";
	int count            = args["--count"]    && args["--count"]->is_integer()   ? args["--count"]->integer()   : 100;
	bool random          = args["--random"]   && args["--random"]->get_boolean();
	
	// Check valid data
	if (start >= end) {
		std::cout << "Invalid interval [" << start << ", " << end << "]" << std::endl;
		return 0;
	}

	if (count <= 0) {
		std::cout << "Invalid count " << count << std::endl;
		return 0;
	}

	// Parse function
	math_func::func* func = math_func::parse(function);
	if (!func) {
		std::cout << "Error parsing function" << std::endl;
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

	// Generate desired set
	std::vector<std::pair<double, double>> set;
	NNSpace::Common::gen_approx_fun(set, [&values, &functions, &opt](double t) {
		values["t"] = t;
		return opt->evaluate(values, functions);
	}, start, end, count, random);
	
	// Write output set
	if (!NNSpace::Common::write_approx_set(set, output))
		std::cout << "Failed output to " << output << std::endl;
	
	return 0;
};