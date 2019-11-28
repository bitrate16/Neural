#!/bin/bash

# compiles & runs comparison between network types in approximation task

set -o xtrace

g++ -O3 src/approx/multistart_cutoff/multistart_cutoff.cpp -o bin/multistart_cutoff -Iinclude -lstdc++fs && ./bin/multistart_cutoff 4 1 3 1 1 10.0 TanH input/train.nse input/test.nse output/mc_network.neetwook

g++ -O3 src/approx/backpropagation/backpropagation.cpp -o bin/backpropagation -Iinclude -lstdc++fs && ./bin/multistart_cutoff 4 1 3 3 1 10.0 TanH input/train.nse input/test.nse output/mc_network.neetwook