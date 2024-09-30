
#!/bin/bash

# Adding jobs to GPUs on hydra:
# addqueue -q gpuQueueYouChose -s --gpus numberOfGPUsYouWant --gputype optionallySpecifyTheType -m CPURamYouNeedInGB./yourProgram
# eg. addqueue -q gpulong -s --gpus 1 --gputype rtx2080with12gb ./myProgram
# eg. addqueue -q gpulong -s -m 4 --gpus 1 ./myProgram
# To look at GPUs active on hydra:
# showgpus

module load cuda
nvcc ../examples/my_model.cu -o exec

addqueue -q gpushort -s --gpus 1 --gputype rtx2080with12gb -m 4 ./exec