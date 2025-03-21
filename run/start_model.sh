#!/bin/bash
# This script is used to compile and run the model on the cluster

# cluster parameters
n_gpus=1
gpu_type="rtx2080with12gb"  # rtx2080with12gb, rtx3090with24gb
queue="gpushort"            # gpushort, gpulong
mem=4                       # allocated memory in GB
comment="3_mins"            # comment for the job

# prepare the environment and load necessary modules
cd /mnt/users/jmalone/GitHub/yalla/run
rm exec*
rm output/*
module load cuda

# generate .h file from default parameters
source ../venv/bin/activate
python3 ../sample/pwriter.py
deactivate

# compile the model
echo "Compiling..."
nvcc -std=c++14 -arch=sm_61 ../examples/eggspot.cu -o exec
echo "Compilation time: $SECONDS seconds"

# remove queue lock file if it exists
if [ -f /mnt/users/jmalone/.addqueuelock ]; then
    addqueue -U
fi

# execute compiled model on cluster and capture console output
output=$(addqueue -c $comment -q $queue -s --gpus $n_gpus --gputype $gpu_type -m $mem ./exec 2>&1)

echo $output
# capture job id from console output
job_id=$(echo "$output" | grep -oP 'exec-\K[0-9]+(?=\.out)')

echo "Submitted Job ID: $job_id"
# wait for the job to complete
while squeue -u $USER | grep $job_id > /dev/null; do
    echo "Job $job_id is still running... Elapsed time: $SECONDS seconds"
    sleep 5  # Check every x seconds
done

echo "Job $job_id has completed."

mv exec-$job_id.out output/exec-$job_id.out 
cp ../examples/eggspot.cu output/eggspot.cu # copy source code into output directory