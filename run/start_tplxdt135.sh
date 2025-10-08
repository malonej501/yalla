#!/bin/bash
# This script is used to compile and run the model on tplxdt135


# prepare the environment and load necessary modules
cd /home/m/malone/GitHub/yalla/run
rm exec*
rm output/*

# generate .h file from default parameters
source ../venv/bin/activate
python3 ../sample/pwriter.py #-p volk_params.csv
deactivate

# compile the model
nvcc -std=c++14 -arch=sm_61 ../examples/eggspot.cu -o exec
# nvcc -std=c++14 -arch=sm_61 ../examples/volk.cu -o exec
# nvcc -std=c++14 -arch=sm_61 ../examples/my_model_2D_volk_birth.cu -o exec

# execute compiled model
./exec

echo "Job $job_id has completed."

cp ../params/default.csv output/default.csv # copy default parameters into output
cp ../examples/eggspot.cu output/eggspot.cu # copy source code into output directory