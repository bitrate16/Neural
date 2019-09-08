#!/bin/bash

# compile <c.sh> "libs" main

clear && gcc -std=c++17 -w -g -Iinclude $2.cpp $1 -lstdc++ -lm -o $2 && ./$2
