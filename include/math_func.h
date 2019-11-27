#pragma once

#include <stdexcept>
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <cmath>

// Represents callable math functions
// a * b - x + sin(cos( e ^ 12))
//
// function ::= <add_exp>
// add_exp  ::= <mul_exp>  | <mul_exp>  + <add_exp> | <mul_exp>  - <add_exp>
// mul_exp  ::= <atomic>   | <atomic> * <mul_exp>   | <atomic> / <mul_exp>  | <atomic> ^ <mul_exp>
// atomic   ::= -<atomic>  | +<atomic> | <name>     | <name>({<add_exp>, }) | <number> | (<add_exp>) | <atomic> ^ <atomic>

namespace math_func {
	typedef std::map<std::string, double> func_constants;
	typedef std::map<std::string, std::function<double(const std::vector<double>&)>> func_functions;
	
	struct func {
		
		virtual ~func() {};
		
		// Should evaluate the function and retuan result
		virtual double evaluate(const func_constants& values, const func_functions& functions) = 0;
		
		virtual void print(std::ostream& os) const = 0;
		
		friend std::ostream& operator<<(std::ostream& os, const func& f);
		
		friend std::ostream& operator<<(std::ostream& os, const func*& f);
		
		virtual func* copy() = 0;
		
		virtual void print(int ident = 0) = 0;
	};
	
	std::ostream& operator<<(std::ostream& os, const func& f) {
		f.print(os);
		
		return os;
	};
	
	std::ostream& operator<<(std::ostream& os, const func*& f) {
		if (!f)
			return os;
		
		f->print(os);
		
		return os;
	};
	
	struct operator_func : public func {
		static const int ADD = 0;
		static const int SUB = 1;
		static const int MUL = 2;
		static const int DIV = 3;
		static const int POW = 4;
		static const int POS = 5;
		static const int NEG = 6;
		
		func* left = nullptr;
		func* right = nullptr;
		int opcode;
		
		operator_func(int _opcode, func* _left = nullptr, func* _right = nullptr) : opcode(_opcode), left(_left), right(_right) {};
		
		~operator_func() {
			if (left) delete left;
			if (right) delete right;
		};
		
		double evaluate(const func_constants& values, const func_functions& functions) {
			switch(opcode) {
				case ADD: return left->evaluate(values, functions) + right->evaluate(values, functions);
				case SUB: return left->evaluate(values, functions) - right->evaluate(values, functions);
				case MUL: return left->evaluate(values, functions) * right->evaluate(values, functions);
				case DIV: return left->evaluate(values, functions) / right->evaluate(values, functions);
				case POW: return std::pow(left->evaluate(values, functions), right->evaluate(values, functions));
				case POS: return + (left->evaluate(values, functions));
				case NEG: return - (left->evaluate(values, functions));
			}
		};
		
		void print(std::ostream& os) const {
			if (ADD <= opcode && opcode <= POW) {
				std::cout << '(';
				if (left) 
					left->print(os);
				else 
					os << "nullptr";
			
				switch(opcode) {
					case ADD: os << " + "; break;
					case SUB: os << " - "; break;
					case MUL: os << " * "; break;
					case DIV: os << " / "; break;
					case POW: os << " ^ "; break;
				}
				
				if (right) 
					right->print(os);
				else 
					os << "nullptr";
				std::cout << ')';
			} else if (POS <= opcode && opcode <= NEG) {
				switch(opcode) {
					case POS: os << '+'; break;
					case NEG: os << '-'; break;
				}
				
				std::cout << '(';
				if (left) 
					left->print(os);
				else 
					os << "nullptr";
				std::cout << ')';
			}
		};
		
		func* copy() {
			return new operator_func(opcode, left ? left->copy() : nullptr, right ? right->copy() : nullptr);
		};
		
		void print(int ident = 0) {
			std::cout << std::string(ident, ' ');
			
			switch(opcode) {
				case ADD: std::cout << '+' << std::endl; break;
				case SUB: std::cout << '-' << std::endl; break;
				case MUL: std::cout << '*' << std::endl; break;
				case DIV: std::cout << '/' << std::endl; break;
				case POW: std::cout << '^' << std::endl; break;
				case POS: std::cout << '+' << std::endl; break;
				case NEG: std::cout << '-' << std::endl; break;
			}
			
			if (ADD <= opcode  && opcode <= POW) {
				if (left) 
					left->print(ident + 1);
				else 
					std::cout << std::string(ident + 1, ' ') << "nullptr" << std::endl;
				
				if (right) 
					right->print(ident + 1);
				else 
					std::cout << std::string(ident + 1, ' ') << "nullptr" << std::endl;
			} else if (POS <= opcode && opcode <= NEG) {
				if (left) 
					left->print(ident + 1);
				else 
					std::cout << std::string(ident + 1, ' ') << "nullptr" << std::endl;
			}
		};
	};
	
	struct name_func : public func {		
		std::string name;
		
		name_func(const std::string& _name) : name(_name) {};
		
		~name_func() {};
		
		double evaluate(const func_constants& values, const func_functions& functions) {
			auto it = values.find(name);
		
			if (it == values.end())
				throw std::invalid_argument("value for " + name + " not defined");
			
			return it->second;
		};
		
		void print(std::ostream& os) const {
			os << name;
		};
		
		func* copy() {
			return new name_func(name);
		};
		
		void print(int ident = 0) {
			std::cout << std::string(ident, ' ') << name << std::endl;
		};
	};
	
	struct const_func : public func {		
		double val;
		
		const_func(double _val = 0) : val(_val) {};
		
		~const_func() {};
		
		double evaluate(const func_constants& values, const func_functions& functions) {
			return val;
		};
		
		void print(std::ostream& os) const {
			os << val;
		};
		
		func* copy() {
			return new const_func(val);
		};
		
		void print(int ident = 0) {
			std::cout << std::string(ident, ' ') << val << std::endl;
		};
	};
	
	struct call_func : public func {		
		std::string name;
		std::vector<func*> args;
		
		call_func(const std::string& _name, const std::vector<func*>& _args) : name(_name), args(_args) {};
		
		~call_func() {
			for (int i = 0; i < args.size(); ++i)
				delete args[i];
		};
		
		double evaluate(const func_constants& values, const func_functions& functions) {
			auto it = functions.find(name);
		
			if (it == functions.end())
				throw std::invalid_argument("function for " + name + " not defined");
			
			std::vector<double> argv(args.size());
			
			for (int i = 0; i < args.size(); ++i)
				argv[i] = args[i]->evaluate(values, functions);
			
			return it->second(argv);
		};
		
		void print(std::ostream& os) const {
			os << name << '(';
			for (int i = 0; i < args.size(); ++i) {
				if (args[i])
					args[i]->print(os);
				else
					os << "nullptr";
				
				if (i != args.size() - 1)
					os << ", ";
			}
			os << ')';
		};
		
		std::vector<func*> copy_args() {
			std::vector<func*> argc(args.size());
			for (int i = 0; i < args.size(); ++i)
				argc[i] = args[i]->copy();
			
			return argc;
		};
		
		func* copy() {			
			return new call_func(name, copy_args());
		};
		
		void print(int ident = 0) {
			std::cout << std::string(ident, ' ') << name << " =>";
			
			for (int i = 0; i < args.size(); ++i) 
				if (args[i]) 
					args[i]->print(ident + 1);
				else 
					std::cout << std::string(ident + 1, ' ') << "nullptr" << std::endl;
		};
	};
	
	namespace parser {
		enum token_type {
			TNAME, TNUM,
			TLBR, TRBR,
			TPLS, TMNS, TMUL, TDIV, TPOW, 
			TCOL, // ','
			TERR
		};
		
		struct token {
			token_type type;
			std::string str;
			double dbl;
			
			token(token_type _type) : type(_type) {};
			token(std::string _str) : type(TNAME), str(_str) {};
			token(double _dbl) : type(TNUM), dbl(_dbl) {};
		};
		
		void print(const std::vector<token>& tokens) {
			for (int i = 0; i < tokens.size(); ++i) {
				switch(tokens[i].type) {
					case TNAME: std::cout << tokens[i].str; break;
					case TNUM : std::cout << tokens[i].dbl; break;
					case TLBR : std::cout << '(';           break;
					case TRBR : std::cout << ')';           break;
					case TPLS : std::cout << '+';           break;
					case TMNS : std::cout << '-';           break;
					case TMUL : std::cout << '*';           break;
					case TDIV : std::cout << '/';           break;
					case TPOW : std::cout << '^';           break;
					case TCOL : std::cout << ',';           break;
					case TERR : std::cout << "TERR ";       break;
				};
			}
			std::cout << std::endl;
		};
		
		std::vector<token> tokenize(const std::string& in) {
			std::vector<token> tokens;
			
			int i = 0;
			
			auto alpha = [](int c) -> bool {
				return 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z' || c == '_';
			};
			
			auto digit = [](int c) -> bool {
				return '0' <= c && c <= '9';
			};
			
			auto white = [](int c) -> bool {
				return c == ' ' || c == '\n' || c == '\r' || c == '\t';
			};
			
			while (1) {
				if (i >= in.size())
					break;
				
				if (white(in[i])) { ++i; continue; }
				
				if (alpha(in[i])) {
					std::string tmp;
					tmp += in[i++];
					
					while (i < in.size() && (alpha(in[i]) || digit(in[i]))) {
						tmp += in[i];
						++i;
					}
					
					tokens.push_back({ tmp });
				} else if (	digit(in[i]) // i, i.
							|| 
							(i < in.size() - 1 // {-,+}i, {-,+}.
								&& 
								(in[i] == '-' 
									|| 
									in[i] == '+') 
								&& 
								(digit(in[i + 1])
									||
									in[i + 1] == '.')
							)
							|| 
							(i < in.size() - 1 // .i
								&& 
								in[i] == '.' 
								&& 
								digit(in[i + 1])
							)
						  ) {
					std::string tmp;
					
					bool has_point = 0;
					
					if (in[i] == '.') {
						has_point = 1;
						tmp += "0.";
						++i;
					} else if (in[i] == '-' || in[i] == '+') {
						tmp += in[i];
						++i;
						
						if (in[i] == '.') {
							has_point = 1;
							++i;
						}
					}
					
					while (i < in.size() && (digit(in[i]) || !has_point && in[i] == '.')) {
						tmp += in[i];
						
						if (in[i] == '.')
							has_point = 1;
						
						++i;
					}
					
					tokens.push_back({ std::stod(tmp) });
				} else if (in[i] == '-') {
					++i;
					tokens.push_back({ TMNS });
				} else if (in[i] == '+') {
					++i;
					tokens.push_back({ TPLS });
				} else if (in[i] == '*') {
					++i;
					tokens.push_back({ TMUL });
				} else if (in[i] == '^') {
					++i;
					tokens.push_back({ TPOW });
				} else if (in[i] == '/') {
					++i;
					tokens.push_back({ TDIV });
				} else if (in[i] == '(') {
					++i;
					tokens.push_back({ TLBR });
				} else if (in[i] == ')') {
					++i;
					tokens.push_back({ TRBR });
				} else {
					tokens.clear();
					tokens.push_back({ TERR });
					return tokens;
				}
			}
			
			return tokens;
		};
		
		func* add_exp(const std::vector<token>& tokens, int& index);
		
		func* atomic(const std::vector<token>& tokens, int& index, int ignore_pow = 0) {
			if (index < tokens.size()) {		
				
				// pow (^) has max priority and should be parsed before other atoms.
				// to avoid recursive loop, add special flag to stop atoms from left-recursion.
				if (!ignore_pow) {
					func* atm = atomic(tokens, index, 1);
					
					if (!atm)
						return nullptr;
					
					if (index >= tokens.size())
						return atm;
					
					if (tokens[index].type == TPOW) {
						++index;
						
						func* atm2 = atomic(tokens, index);
						
						if (!atm2) {
							delete atm;
							return nullptr;
						}
						
						return new operator_func(operator_func::POW, atm, atm2);
					}
					
					return atm;
				}
				
				if (tokens[index].type == TNAME) { // name, name(...)
					std::string name = tokens[index].str;
					++index;
					
					if (index < tokens.size() && tokens[index].type == TLBR) {						
						++index;
						
						std::vector<func*> args;
						
						while (1) {
							// Check EOF
							if (index >= tokens.size()) {
								for (int i = 0; i < args.size(); ++i)
									delete args[i];
								return nullptr;
							}
							
							// Check ')'
							if (tokens[index].type == TRBR) {
								++index;
								break;
							}
							
							// Read arg[i]
							func* exp2 = add_exp(tokens, index);
							if (!exp2) {
								for (int i = 0; i < args.size(); ++i)
									delete args[i];
								return nullptr;
							}
							args.push_back(exp2);
							
							// Check EOF
							if (index >= tokens.size()) {
								for (int i = 0; i < args.size(); ++i)
									delete args[i];
								return nullptr;
							}
							
							// Check ')'
							if (tokens[index].type == TRBR) {
								++index;
								break;
							}
							
							// Check ','
							if (tokens[index].type != TCOL) {
								for (int i = 0; i < args.size(); ++i)
									delete args[i];
								return nullptr;
							} else
								++index;
						}
						
						return new call_func(name, args);
					} else 
						return new name_func(name);
				}
				
				if (tokens[index].type == TPLS) { // +exp
					++index;
					func* exp2 = atomic(tokens, index);
					
					if (!exp2) 
						return nullptr;
					
					return new operator_func(operator_func::POS, exp2);
				}
				
				if (tokens[index].type == TMNS) { // -exp
					++index;
					func* exp2 = atomic(tokens, index);
					
					if (!exp2) 
						return nullptr;
					
					return new operator_func(operator_func::NEG, exp2);
				}
				
				if (tokens[index].type == TNUM) { // num
					++index;
					
					return new const_func(tokens[index - 1].dbl);
				}
				
				if (tokens[index].type == TLBR) { // (...)
					++index;
					
					func* brk = add_exp(tokens, index);
					
					if (index >= tokens.size() || tokens[index].type != TRBR) {
						delete brk;
						return nullptr;
					} else
						++index;
					
					return brk;
				}
				
				return nullptr;
			} else
				return nullptr;
		};
		
		func* mul_exp(const std::vector<token>& tokens, int& index) {
			func* exp1 = atomic(tokens, index);
			
			if (!exp1)
				return nullptr;
			
			if (index < tokens.size()) {				
				if (tokens[index].type == TMUL) {
					++index;
					func* exp2 = mul_exp(tokens, index);
					
					if (!exp2) {
						delete exp1;
						return nullptr;
					}
					
					return new operator_func(operator_func::MUL, exp1, exp2);
				}
				
				if (tokens[index].type == TDIV) {
					++index;
					func* exp2 = mul_exp(tokens, index);
					
					if (!exp2) {
						delete exp1;
						return nullptr;
					}
					
					return new operator_func(operator_func::DIV, exp1, exp2);
				}
				
				return exp1;
			} else
				return exp1;
		};
		
		func* add_exp(const std::vector<token>& tokens, int& index) { 
			func* exp1 = mul_exp(tokens, index);
			
			if (!exp1)
				return nullptr;
			
			if (index < tokens.size()) {
				if (tokens[index].type == TPLS) {
					++index;
					func* exp2 = add_exp(tokens, index);
					
					if (!exp2) {
						delete exp1;
						return nullptr;
					}
					
					return new operator_func(operator_func::ADD, exp1, exp2);
				}
				
				if (tokens[index].type == TMNS) {
					++index;
					func* exp2 = add_exp(tokens, index);
					
					if (!exp2) {
						delete exp1;
						return nullptr;
					}
					
					return new operator_func(operator_func::SUB, exp1, exp2);
				}
				
				return exp1;
			} else
				return exp1;
		};
	
		func* parse(const std::string& in) {
			int index = 0;
			std::vector<token> tokens = tokenize(in);
			
			if (tokens.size() == 1 && tokens[0].type == TERR)
				return nullptr;
			
			func* exp = add_exp(tokens, index);
			return exp;
		};
	};
	
	// Does parsing the input string and returns resulting function.
	// On error ocurred, returns nullptr.
	static func* parse(const std::string& in) {
		return parser::parse(in);
	};
};