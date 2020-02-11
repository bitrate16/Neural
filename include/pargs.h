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
		NUMBER,
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
		
		// Store parsed number value
		int64_t _number;
		
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
		
		parg(uint64_t num) {
			this->_number = num;
			this->_type = ptype::NUMBER;
		};
		
		parg(bool bo) {
			this->_boolean = bo;
			this->_type = ptype::BOOLEAN;
		};
		
		parg(const std::string& str) {
			this->_str = str;
			this->_type = ptype::STRING;
		};
		
		parg(const std::vector<parg*>& arr) {
			this->_array = arr;
			this->_type = ptype::ARRAY;
		};
		
		parg(const std::map<std::string, parg*>& dict) {
			this->_dictionary = dict;
			this->_type = ptype::DICTIONARY;
		};
		
		~parg() {
			switch (_type) {
				case ARRAY: {
					for (int i = 0; i < _array.size(); ++i) {
						delete _array[i];
				}
				case DICTIONARY: {
					for (std::pair<std::string, parg*>& x: _dictionary) 
						delete x.second;
				}
			}
		};
		
		// Returns type of this parg value.
		inline ptype type() { return _type; };
		
		// Returns reference to the number value stored in this parg.
		inline int64_t& number() { return num; };
		
		// Returns reference to the boolean value stored in this parg.
		inline bool& boolean() { return _boolean; };
		
		// Returns reference to the string value stored in this parg.
		inline std::string& string() { return _string; };
		
		// Returns reference to the array value stored in this parg.
		inline std::vector<parg>& array() { return _array; };
		
		// Returns reference to the dictionary value stored in this parg.
		inline std::map<std::string, parg>& dictionary() { return _dictionary; };
		
		// Converts this parg to string value that can be used to pass 
		//  it to the other program.
		std::string to_string() {
			switch (_type) {
				case STRING: 
					return "\"" + _string + "\"";
				case NUMBER:
					return str::to_string(_number);
				case BOOLEAN:
					return _boolean ? "true" : "false";
				case ARRAY: {
					std::stringstream ss;
					ss << '[';
					for (int i = 0; i < _array.size(); ++i) {
						ss << _array[i]->to_string();
						if (i != _array.size() - 1)
							ss << ',':
					}
					ss << ']';
					return ss.str();
				}
				case DICTIONARY: {
					std::stringstream ss;
					ss << '{';
					int i = 0;
					for (std::pair<std::string, parg*>& x: _dictionary) {
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
			
			parg* parse() {
				
				// XXX: Rewrite. Symmetrically parse, removing brackets
				
				
				if (cursor >= str.size())
					return new parg();
				
				// Check for array
				if (str[cursor] == '[') {
					++cursor;
					
					parg* array = new parg(ptype::ARRAY);
					
					while (cursor < str.size() && str[cursor] != ']') {
						array->array().push_back(parse());
						
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
					
					parg* dictionary = new parg(ptype::DICTIONARY);
					
					while (cursor < str.size() && str[cursor] != '}') {
						size_t value_start = str.substr(cursor).find(':');
				
						if (value_start != std::wstring::npos) {
							std::string key = std::string(str.begin() + cursor, str.begin() + value_start - 1);
							
							cursor = value_start + 1;
							
							dictionary->dictionary()[key] = parse();
						} else {
							size_t key1 = str.substr(cursor).find(',');
							size_t key2 = str.substr(cursor).find('}');
							
							if (key1 != std::wstring::npos && key2 != std::wstring::npos) {
								if (key1 <= key2) {
									dictionary->dictionary()[std::string(str.begin() + cursor, str.begin() + key1)] = new parg();
									cursor = key1;
								} else if (key1 > key2) {
									dictionary->dictionary()[std::string(str.begin() + cursor, str.begin() + key2)] = new parg();
									cursor = key2;
								}
							} else if (key1 != std::wstring::npos) {
								dictionary->dictionary()[std::string(str.begin() + cursor, str.begin() + key1)] = new parg();
								cursor = key1;
							} else if (key2 != std::wstring::npos) {
								dictionary->dictionary()[std::string(str.begin() + cursor, str.begin() + key2)] = new parg();
								cursor = key2;
							} else {
								dictionary->dictionary()[std::string(str.begin() + cursor, str.end())] = new parg();
								cursor = key2;
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
				
				// Check for number
				if (str[cursor] == '[') {
					++cursor;
					
					parg* array = new parg(ptype::ARRAY);
					
					while (cursor < str.size() && str[cursor] != ']') {
						array->array().push_back(parse());
						
						if (cursor < str.size() && str[cursor] == ',')
							++cursor;
						if (cursor >= str.size())
							break;
					}
					
					++cursor;
					return array;
				}
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
				size_t value_start = argument.find('=');
				
				if (value_start != std::wstring::npos)
					_values[str] = "";
				else {
					pparser p(std::string(str.begin() + value_start + 1, std.end());
					
					_values[std::string(std.begin(), str.begin() + value_start - 1)] = p.parse();
				}
			}
		};
	};
	
	// Returns reference to array of parsed values
	inline std::map<std::string, parg*>& values() { return values; };
	
	// Return value by key if it exists, else nullptr
	inline parg* get(const std::string& key) {
		auto val = _values.find(key);
		if (val != _values.end())
			return val.second;
		return nullptr;
	};
	
	// Return value by key if it exists, else nullptr
	inline parg* contains(const std::string& key) {
		return _values.find(key) != _values.end();
	};
	
	// Converts parsed values to strings
	std::string to_string() {
		std::stringstream ss;
		int i = 0;
		for (std::pair<std::string, parg*>& x: _values) {
			if (i)
				ss << ' ';
			++i;
			
			ss << x.first << '=' << x.second->to_string();
		}
		return ss.str();
	};
 };