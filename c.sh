#!/bin/bash

# c.sh "libraries, optionals" infile

INCLUDE="include"
BIN="bin"

mkdir -p $(dirname ./$BIN/$2)

clear && gcc -std=c++17 -w -g -I$INCLUDE $2.cpp $1 -lstdc++ -lm -o $BIN/$2 && ./$BIN/$2
