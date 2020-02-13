#pragma once

#include <map>
#include <vector>
#include <string>
#include <sstream>
#include <cstdint>

/* 
 * (PARGS) Parse ARGS utility.
 *  - Minimal syntax arguments parsing utility.
 * 
 * Supports parsing of input program arguments
 *  in format:
 *  -<char>; --<sequence>; -<char>=<expression>; 
 *  --<sequence>=<expression>; <sequence>.
 *  
 * <expression> can be simple string value or a 
 *  complex expression.
 * 
 * Definition:
 * 
 * <expression> ::= <string> | <array> | <number>
 *                | <boolean> | <dictionary>
 * <boolean> ::= true | false
 * <string> ::= <any acceptable char sequence>
 * <array> ::= [ [[<expression>],] ]
 * <number> ::= <any integer notation>
 * <dictionary> ::= { [[<string> : <expression>],] }
 * 
 * Example:
 * 
 * -a -o=foo --vla={key:any,name:apply} --ignore-all foo.txt
 * 
 */ 

 namespace pargs {
	enum ptype {
		STRING,
		BOOLEAN,
		INTEGER,
		REAL,
		ARRAY,
		DICTIONARY
	};
	
	/*
	 * Represents single value entry.
	 * Can handle value of the argument 
	 *  or subscript of an array.
	 */
	class parg {
		friend class pargs;
		friend class pparser;
		
		// Store parsed value of the dictionary.
		std::map<std::string, parg*> _dictionary;
		
		// Store parsed value of the array.
		std::vector<parg*> _array;
		
		// Store parsed integer value
		int64_t _integer;
		
		// Store parsed double value
		double _real;
		
		// Store parsed boolean value
		bool _boolean;
		
		// Store string value
		std::string _string;
		
		// Store value type
		ptype _type = ptype::STRING;
		
		// Construct with type
		parg(ptype t) {
			this->_type = t;
		};
		
	public:
		
		// Constructors.
		// Public to allow used construct his own arguments.
		
		parg() {
			this->_type = ptype::STRING;
		};
		
		~parg() {
			switch (_type) {
				case ARRAY: {
					for (int i = 0; i < _array.size(); ++i)
						delete _array[i];
				}
				case DICTIONARY: {
					for (auto& x: _dictionary) 
						delete x.second;
				}
			}
		};
		
		inline static parg* Integer(int64_t i) {
			parg* p  = new parg();
			p->_type = ptype::INTEGER;
			p->_integer = i;
			return p;
		};
		
		inline static parg* Real(double d) {
			parg* p  = new parg();
			p->_type = ptype::REAL;
			p->_real = d;
			return p;
		};
		
		inline static parg* Boolean(bool i) {
			parg* p  = new parg();
			p->_type = ptype::BOOLEAN;
			p->_boolean = i;
			return p;
		};
		
		inline static parg* String(const std::string& i) {
			parg* p  = new parg();
			p->_type = ptype::STRING;
			p->_string = i;
			return p;
		};
		
		inline static parg* Array(const std::vector<parg*>& i) {
			parg* p  = new parg();
			p->_type = ptype::ARRAY;
			p->_array = i;
			return p;
		};
		
		inline static parg* Dictionary(const std::map<std::string, parg*>& i) {
			parg* p  = new parg();
			p->_type = ptype::DICTIONARY;
			p->_dictionary = i;
			return p;
		};
		
		// Returns type of this parg value.
		inline ptype type() { return _type; };
		
		// Returns reference to the integer value stored in this parg.
		inline int64_t& integer() { return _integer; };
		
		inline bool is_integer() { return _type == ptype::INTEGER; };
		
		inline int64_t get_integer() { return is_integer() ? _integer : _real; };
		
		// Returns reference to the double value stored in this parg.
		inline double& real() { return _real; };
		
		inline bool is_real() { return _type == ptype::REAL; };
		
		inline double get_real() { return is_integer() ? _integer : _real; };
		
		inline bool is_number() { return _type == ptype::INTEGER || _type == ptype::REAL; };
		
		// Returns reference to the boolean value stored in this parg.
		inline bool& boolean() { return _boolean; };
		
		inline bool is_boolean() { return _type == ptype::BOOLEAN; };
		
		// Returns reference to the string value stored in this parg.
		inline std::string& string() { return _string; };
		
		inline bool is_string() { return _type == ptype::STRING; };
		
		// Returns reference to the array value stored in this parg.
		inline std::vector<parg*>& array() { return _array; };
		
		inline bool is_array() { return _type == ptype::ARRAY; };
		
		// Returns reference to the dictionary value stored in this parg.
		inline std::map<std::string, parg*>& dictionary() { return _dictionary; };
		
		inline bool is_dictionary() { return _type == ptype::DICTIONARY; };
		
		// Converts this parg to string value that can be used to pass 
		//  it to the other program.
		std::string to_string() {
			switch (_type) {
				case STRING: 
					return "\"" + _string + "\"";
				case INTEGER:
					return std::to_string(_integer);
				case REAL:
					return std::to_string(_real);
				case BOOLEAN:
					return _boolean ? "true" : "false";
				case ARRAY: {
					std::stringstream ss;
					ss << '[';
					for (int i = 0; i < _array.size(); ++i) {
						ss << _array[i]->to_string();
						if (i != _array.size() - 1)
							ss << ',';
					}
					ss << ']';
					return ss.str();
				}
				case DICTIONARY: {
					std::stringstream ss;
					ss << '{';
					int i = 0;
					for (auto& x: _dictionary) {
						if (i)
							ss << ',';
						++i;
						
						ss << x.first << ':' << x.second->to_string();
					}
					ss << '}';
					return ss.str();
				}
			}
		};
	};
	
	// XXX: Parse
	class pargs {
		// Used for parsing input string int objects.
		class pparser {
			friend class pargs;
			
			int         cursor = 0;
			std::string str;
			
			pparser(const std::string& s) {
				str = s;
			}
			
			// Parse until reaching:
			// 0 - nothing
			// 1 - , or ] in array parse
			// 2 - , or } in dictionary parse
			parg* parse(int parent_object = 0) {				
				
				if (cursor >= str.size())
					return new parg();
				
				// Check for array
				if (str[cursor] == '[') {
					++cursor;
					
					parg* array = new parg();
					array->_type = ptype::ARRAY;
					
					while (cursor < str.size() && str[cursor] != ']') {
						array->array().push_back(parse(1));
						
						if (cursor < str.size() && str[cursor] == ',')
							++cursor;
						if (cursor >= str.size())
							break;
					}
					
					++cursor;
					return array;
				}
				
				// Check for dictionary
				if (str[cursor] == '{') {
					++cursor;
					
					parg* dictionary = new parg();
					dictionary->_type = ptype::DICTIONARY;
					
					while (cursor < str.size() && str[cursor] != '}') {
						size_t value_start = str.substr(cursor).find(':');
				
						if (value_start != std::wstring::npos) {
							std::string key = std::string(str.begin() + cursor, str.begin() + cursor + value_start);
							
							cursor += value_start + 1;
							
							dictionary->dictionary()[key] = parse(2);
						} else {
							size_t key1 = str.substr(cursor).find(',');
							size_t key2 = str.substr(cursor).find('}');
							
							if (key1 != std::wstring::npos && key2 != std::wstring::npos) {
								if (key1 <= key2) {
									dictionary->dictionary()[std::string(str.begin() + cursor, str.begin() + cursor + key1)] = new parg();
									cursor += key1;
								} else if (key1 > key2) {
									dictionary->dictionary()[std::string(str.begin() + cursor, str.begin() + cursor + key2)] = new parg();
									cursor += key2;
								}
							} else if (key1 != std::wstring::npos) {
								dictionary->dictionary()[std::string(str.begin() + cursor, str.begin() + cursor + key1)] = new parg();
								cursor += key1;
							} else if (key2 != std::wstring::npos) {
								dictionary->dictionary()[std::string(str.begin() + cursor, str.begin() + cursor + key2)] = new parg();
								cursor += key2;
							} else {
								dictionary->dictionary()[std::string(str.begin() + cursor, str.end())] = new parg();
								cursor = str.size();
							}
						}
						
						if (cursor < str.size() && str[cursor] == ',')
							++cursor;
						if (cursor >= str.size())
							break;
					}
					
					++cursor;
					return dictionary;
				}
				
				// Read full string
				int ind = cursor;
				for (; ind < str.size(); ++ind) {
					if (parent_object == 1 && (str[ind] == ',' || str[ind] == ']'))
						break;
					if (parent_object == 2 && (str[ind] == ',' || str[ind] == '}'))
						break;
				}
				
				std::string strv(str.begin() + cursor, str.begin() + ind);
				cursor = ind;
				
				// Check for boolean
				if (strv == "true")
					return parg::Boolean(true);
				if (strv == "false")
					return parg::Boolean(false);
				
				// Check for number
				bool is_real    = 0;
				double real     = 0;
				bool is_int     = 0;
				int64_t integer = 0;
				
				auto stream = std::istringstream(strv);
				stream >> real;      
				is_real = !stream.fail() && stream.eof();
				
				stream = std::istringstream(strv);
				stream >> integer;      
				is_int = !stream.fail() && stream.eof();
				
				if (is_real && !is_int)
					return parg::Real(real);
				if (is_real && is_int)
					return parg::Integer((int64_t) integer);
				
				return parg::String(strv);
			};
		};

		std::map<std::string, parg*> _values;
		
	public:
		
		// Accepts input arguments of the program and 
		//  performs simple parsing into objects.
		pargs(int argc, const char** argv) {
			// Perform parsing input strings
			for (int i = 0; i < argc; ++i) {
				std::string str = argv[i];
				
				// Try parse =
				size_t value_start = str.find('=');
				
				if (_values.find(str) != _values.end())
					delete _values[str];
				
				if (value_start == std::wstring::npos)
					_values[str] = new parg();
				else {
					pparser p(std::string(str.begin() + value_start + 1, str.end()));
					
					_values[std::string(str.begin(), str.begin() + value_start)] = p.parse();
				}
			}
		};
	
		// Returns reference to array of parsed values
		inline std::map<std::string, parg*>& values() { return _values; };
		
		// Return value by key if it exists, else nullptr
		inline parg* get(const std::string& key) {
			auto val = _values.find(key);
			if (val != _values.end())
				return val->second;
			return nullptr;
		};
		
		// Return value by key if it exists, else nullptr
		inline bool contains(const std::string& key) {
			return _values.find(key) != _values.end();
		};
		
		// Converts parsed values to strings
		std::string to_string() {
			std::stringstream ss;
			int i = 0;
			for (auto& x: _values) {
				if (i)
					ss << ' ';
				++i;
				
				ss << x.first << '=' << x.second->to_string();
			}
			return ss.str();
		};
	};
 };