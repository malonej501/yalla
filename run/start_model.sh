#!/bin/bash

cd /mnt/users/jmalone/GitHub/yalla/run
rm exec*
rm output/*
module load cuda
# nvcc -std=c++14 -arch=sm_61 ../examples/my_model_2D.cu -o exec
nvcc -std=c++14 -arch=sm_61 ../examples/eggspot.cu -o exec
# nvcc -std=c++14 -arch=sm_61 ../examples/eggspot_one_cell_type.cu -o exec
# nvcc -std=c++14 -arch=sm_61 ../examples/eggspot_simple_force_potential.cu -o exec
# nvcc -std=c++14 -arch=sm_61 ../examples/my_model_2D_reduced.cu -o exec
# nvcc -std=c++14 -arch=sm_61 ../examples/my_model_2D_volk_birth.cu -o exec

# #addqueue -q gpushort -s --gpus 1 --gputype rtx2080with12gb -m 4 ./exec
# if not [ ! -f/mnt/users/jmalone/.addqueuelock ]; then
#     addqueue -U
# fi
# capture console output
output=$(addqueue -c "3_mins" -q gpushort -s --gpus 1 --gputype rtx2080with12gb -m 4 ./exec 2>&1)
# output=$(addqueue -c "3_mins" -q gpulong -s --gpus 1 --gputype rtx3090with24gb -m 4 ./exec 2>&1)


echo $output

job_id=$(echo "$output" | grep -oP 'exec-\K[0-9]+(?=\.out)')

echo "Submitted Job ID: $job_id"
# wait for the job to complete
while squeue -u $USER | grep $job_id > /dev/null; do
    echo "Job $job_id is still running... Elapsed time: $SECONDS seconds"
    sleep 5  # Check every 10 seconds
done

echo "Job $job_id has completed."

mv exec-$job_id.out output/exec-$job_id.out 

# copy the output folder to the local machine
# scp -r output/ $(echo $SSH_CONNECTION | cut -f 1 -d ' '):/home/jmalone/GitHub/yalla/run/saves