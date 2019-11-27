#pragma once

#include "math_func.h"

namespace math_func {
	
	func* mul(func* a, func* b) { 
		return new operator_func(operator_func::MUL, a, b);
	};
	
	func* div(func* a, func* b) { 
		return new operator_func(operator_func::DIV, a, b);
	};
	
	func* sum(func* a, func* b) { 
		return new operator_func(operator_func::ADD, a, b);
	};
	
	func* sub(func* a, func* b) { 
		return new operator_func(operator_func::SUB, a, b);
	};
	
	func* neg(func* a) { 
		return new operator_func(operator_func::NEG, a);
	};
	
	func* pow(func* a, func* b) { 
		return new operator_func(operator_func::POW, a, b);
	};
	
	func* pow(func* a, double value) { 
		return new operator_func(operator_func::POW, a, new const_func(value));
	};
	
	// Returns new func with value of derivative with respect to varname
	func* derivate(func* f, const std::string& varname) {
		if (f == nullptr)
			throw std::runtime_error("derivate failed: nullptr");
		
		if (dynamic_cast<const_func*>(f)) // const' = 0
			return new const_func(0);
			
		if (dynamic_cast<name_func*>(f)) { // x' = 1
			name_func* t = dynamic_cast<name_func*>(f);
			if (t->name == varname)
				return new const_func(1);
			else
				return new const_func(0);
		}
		
		if (dynamic_cast<call_func*>(f)) {
			call_func* t = dynamic_cast<call_func*>(f);
			
			// d f(x, y, z, ...) / dt = f'[x] (x, y, z, ...) * x'[t] + f'[y] (x, y, z, ...) * y'[t] + ...
			
			if (t->name == "sin") {
				if (t->args.size() != 1)
					throw std::runtime_error("derivate failed: cos(x)");
				
				func* fun = new call_func("cos", t->copy_args());
				
				func* der = mul(fun, derivate(t->args[0], varname));
				
				// for (int i = 1; i < t->args.size(); ++i)
				// 	der = sum(der, mul(fun->copy(), derivate(t->args[i], varname)));
				
				return der;
			}
			
			if (t->name == "cos") {
				if (t->args.size() != 1)
					throw std::runtime_error("derivate failed: cos(x)");
				
				func* fun = neg(new call_func("sin", t->copy_args()));
				
				func* der = mul(fun, derivate(t->args[0], varname));
				
				return der;
			}
			
			if (t->name == "ln") {
				if (t->args.size() != 1)
					throw std::runtime_error("derivate failed: ln(x)");
				
				return div(derivate(t->args[0], varname), t->args[0]->copy());
			}
			
			throw std::runtime_error("derivate function failed: function not defined");
		}
		
		if (dynamic_cast<operator_func*>(f)) {
			operator_func* t = dynamic_cast<operator_func*>(f);
			
			// a + b -> a' + b'
			// a - b -> a' - b'
			// a * b -> a' * b + a * b'
			// a / b -> (a' * b - a * b') / b ^ 2
			// a ^ b -> a' * b * a ^ (b - 1)
			// -a -> -a'
			// +a -> +a'
			
			switch(t->opcode) {
				case operator_func::ADD:
					return sum(derivate(t->left, varname), derivate(t->right, varname));
				case operator_func::SUB: 
					return sub(derivate(t->left, varname), derivate(t->right, varname));
				case operator_func::MUL: 
					return  sum(mul(derivate(t->left, varname), t->right->copy()), 
								mul(t->left->copy(), derivate(t->right, varname)));
				case operator_func::DIV: 
					return  div(sub(mul(derivate(t->left, varname), t->right->copy()), 
									mul(t->left->copy(), derivate(t->right, varname))),
								pow(t->right->copy(), 2.0));
				case operator_func::POW: 
					return  mul(derivate(t->left, varname),
								mul(t->right->copy(),
									pow(t->left->copy(),
										sub(t->right->copy(),
											new const_func(1)))));
				case operator_func::POS:
					return derivate(t->left, varname);
				case operator_func::NEG: 
					return neg(derivate(t->left, varname));
				default:
					throw std::runtime_error("derivate failed: undefined operation");
			}
		}
			
		throw std::runtime_error("derivate failed: undefined");
	};

	// Performs calculation optimization by collecting trailing constants
	// ex: x + 1 + 1 + 1 + 0 -> x + 3
	func* optimize(func* f) {
		if (f == nullptr)
			throw std::runtime_error("optimize failed: nullptr");
		
		if (dynamic_cast<const_func*>(f))
			return f->copy();
		
		if (dynamic_cast<name_func*>(f)) 
			return f->copy();
		
		if (dynamic_cast<call_func*>(f)) {
			call_func* t = dynamic_cast<call_func*>(f);
			
			std::vector<func*> opt_args(t->args.size());
			
			for (int i = 0; i < t->args.size(); ++i)
				opt_args[i] = optimize(t->args[i]);
			
			return new call_func(t->name, opt_args);
		}
		
		if (dynamic_cast<operator_func*>(f)) {
			operator_func* t = dynamic_cast<operator_func*>(f);
			
			if (operator_func::ADD <= t->opcode && t->opcode <= operator_func::POW) {
				func* a = optimize(t->left);
				func* b = optimize(t->right);
				
				const_func* p;
				const_func* q;
				
				if (a && b && (p = dynamic_cast<const_func*>(a)) && (q = dynamic_cast<const_func*>(b))) {
					func* n;
					
					if (t->opcode == operator_func::ADD) n = new const_func(p->val + q->val);
					if (t->opcode == operator_func::SUB) n = new const_func(p->val - q->val);
					if (t->opcode == operator_func::MUL) n = new const_func(p->val * q->val);
					if (t->opcode == operator_func::DIV) n = new const_func(p->val / q->val);
					if (t->opcode == operator_func::POW) n = new const_func(std::pow(p->val, q->val));
					
					delete a;
					delete b;
					
					return n;
				} else if (t->opcode == operator_func::MUL) {
					if (a && (p = dynamic_cast<const_func*>(a)) && p->val == 0) { // 0 * exp
						delete a;
						delete b;
						return new const_func(0);
					} else if (a && (p = dynamic_cast<const_func*>(a)) && p->val == 1) { // 1 * exp
						delete a;
						return b;
					} else if (b && (q = dynamic_cast<const_func*>(b)) && q->val == 0) { // exp * 0
						delete a;
						delete b;
						return new const_func(0);
					} else if (b && (q = dynamic_cast<const_func*>(b)) && q->val == 1) { // exp * 1
						delete b;
						return a;
					} else
						return new operator_func(t->opcode, a, b);
				} else if (t->opcode == operator_func::DIV) {
					if (a && (p = dynamic_cast<const_func*>(a)) && p->val == 0) { // 0 / exp
						delete a;
						delete b;
						return new const_func(0);
					} else
						return new operator_func(t->opcode, a, b);
				} else if (t->opcode == operator_func::POW) {
					if (b && (q = dynamic_cast<const_func*>(b)) && q->val == 0) { // exp ^ 0
						delete a;
						delete b;
						return new const_func(1);
					} else if (b && (q = dynamic_cast<const_func*>(b)) && q->val == 1) { // exp ^ 1
						delete b;
						return a;
					} else
						return new operator_func(t->opcode, a, b);
				} else if (t->opcode == operator_func::ADD) {
					if (a && (p = dynamic_cast<const_func*>(a)) && p->val == 0) { // 0 + exp
						delete a;
						return b;
					} else if (b && (q = dynamic_cast<const_func*>(b)) && q->val == 0) { // exp + 0
						delete b;
						return a;
					} else
						return new operator_func(t->opcode, a, b);
				} else if (t->opcode == operator_func::SUB) {
					if (a && (p = dynamic_cast<const_func*>(a)) && p->val == 0) { // 0 - exp
						delete a;
						
						operator_func* o;
						
						if ((o = dynamic_cast<operator_func*>(b)) && o->opcode == operator_func::NEG) { // - -exp -> exp
							func* r = o->left;
							o->left = nullptr;
							delete b;
							
							return r;
						} else
							return neg(b);
						
					} else if (b && (q = dynamic_cast<const_func*>(b)) && q->val == 0) { // exp - 0
						delete b;
						return a;
					} else
						return new operator_func(t->opcode, a, b);
				} else
					return new operator_func(t->opcode, a, b);
				
			} else if (operator_func::POS <= t->opcode  && t->opcode <= operator_func::NEG) {
				if (t->left && dynamic_cast<const_func*>(t->left))
					return new const_func(t->opcode == operator_func::NEG ? -dynamic_cast<const_func*>(t->left)->val : dynamic_cast<const_func*>(t->left)->val);
				else if (!t->left) {
					throw std::runtime_error("optimize failed: nullptr");
				} else {
					func* n = optimize(t->left);
					
					if (dynamic_cast<const_func*>(n)) {
						func* p = new const_func(t->opcode == operator_func::NEG ? -dynamic_cast<const_func*>(n)->val : dynamic_cast<const_func*>(n)->val);
						delete n;
						return p;
					} else 
						return t->opcode == operator_func::NEG ? neg(n) : n;
				}
			} else
				throw std::runtime_error("optimize failed: undefined operation");
		}
		
		throw std::runtime_error("optimize failed: undefined func");
	};

	// Numerical integration
	// Passed values table is used for insertation of integration variable value.
	double num_integrate(func* f, func_constants& values, const func_functions& functions, const std::string& varname, double a, double b, long iterations) {
		if (!f)
			throw std::runtime_error("integrate failed: nullptr");
		
		bool sign = 0;
		if (b < a) {
			sign = 1;
			double tmp = a;
			a = b;
			b = tmp;
		}
		
		double dt = (b - a) / static_cast<double>(iterations);
		double integral = 0.0;
		
		if (dt == 0.0)
			return 0.0;
		
		for (double t = a; t <= b + dt; t += dt) {
			values[varname] = t;
			integral += f->evaluate(values, functions);
		}
		integral *= dt;
		
		return sign ? -integral : integral;
	};
};