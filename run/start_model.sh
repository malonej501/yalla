
#!/bin/bash


rm exec*
rm output/*
module load cuda
nvcc -std=c++14 -arch=sm_61 ../examples/my_model_3D.cu -o exec

addqueue -q gpushort -s --gpus 1 --gputype rtx2080with12gb -m 4 ./exec
# addqueue -q gpushort -s --gpus 1 --gputype rtx2080with12gb -m 4 