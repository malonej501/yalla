
#!/bin/bash

module load cuda
nvcc ../examples/my_model.cu -o exec

addqueue -q gpushort -s --gpus 1 --gputype rtx2080with12gb -m 4 ./exec