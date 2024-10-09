#!/bin/bash

rm exec*
rm output/*
module load cuda
nvcc -std=c++14 -arch=sm_61 ../examples/my_model_2D.cu -o exec

#addqueue -q gpushort -s --gpus 1 --gputype rtx2080with12gb -m 4 ./exec
# capture console output
output=$(addqueue -c "3_mins" -q gpushort -s --gpus 1 --gputype rtx2080with12gb -m 4 ./exec 2>&1)


echo $output

job_id=$(echo "$output" | grep -oP 'exec-\K[0-9]+(?=\.out)')

echo "Submitted Job ID: $job_id"
# wait for the job to complete
while squeue -u $USER | grep $job_id > /dev/null; do
    echo "Job $job_id is still running... Elapsed time: $SECONDS seconds"
    sleep 5  # Check every 10 seconds
done

echo "Job $job_id has completed."

# copy the output folder to the local machine
# scp -r output/ $(echo $SSH_CONNECTION | cut -f 1 -d ' '):/home/jmalone/GitHub/yalla/run/saves