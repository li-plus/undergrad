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
bin/PA2 testcases/scene01_basic.txt
bin/PA2 testcases/scene02_cube.txt
bin/PA2 testcases/scene03_sphere.txt
bin/PA2 testcases/scene04_axes.txt
bin/PA2 testcases/scene05_bunny_200.txt
bin/PA2 testcases/scene06_bunny_1k.txt
bin/PA2 testcases/scene07_shine.txt
