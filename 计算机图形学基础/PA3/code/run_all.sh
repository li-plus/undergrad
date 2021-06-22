#!/usr/bin/env bash

# If project not ready, generate cmake file.
if [[ ! -d build ]]; then
    mkdir -p build
    cd build
    cmake ..
    cd ..
else
    rm -r build
    mkdir -p build
    cd build
    cmake ..
    cd ..
fi

# Build project.
cd build
make -j
cd ..

# Run all testcases. 
# You can comment some lines to disable the run of specific examples.
bin/PA3 testcases/scene01_basic.txt
bin/PA3 testcases/scene04_axes.txt
bin/PA3 testcases/scene06_bunny_1k.txt
bin/PA3 testcases/scene08_core.txt
bin/PA3 testcases/scene09_norm.txt
bin/PA3 testcases/scene10_wineglass.txt
