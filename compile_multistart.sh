#!/bin/bash

# OPTIONAL="-DENABLE_PRINT"
OPTIONAL=$1

set -o xtrace

g++ -O3 $OPTIONAL src/multistart_cutoff/generate_networks.cpp -o bin/generate_networks -Iinclude -lstdc++fs
g++ -O3 $OPTIONAL src/multistart_cutoff/train_network.cpp -o bin/train_network -Iinclude -lstdc++fs
g++ -O3 $OPTIONAL src/split_set.cpp -o bin/split_set -Iinclude -lstdc++fs
g++ -O3 $OPTIONAL src/multistart_cutoff/calculate_error.cpp -o bin/calculate_error -Iinclude -lstdc++fs
