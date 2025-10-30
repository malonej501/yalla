#!/bin/bash
# This script is used to compile and run the model on the cluster

# cluster parameters
n_gpus=1
gpu_type="rtx2080with12gb"  # rtx2080with12gb, rtx3090with24gb
queue="gpulong"            # gpushort, gpulong
mem=4                       # allocated memory in GB
comment="3_mins"            # comment for the job

POSITIONAL_ARGS=()      # array to hold positional arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -v|--version)
        version="$2"
        shift # past argument
        shift # past value
        ;;
    *)    # unknown option
        POSITIONAL_ARGS+=("$1") # save it in an array for later
        shift # past argument
        ;;
  esac
done
set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters
echo "Version: ${version:-default}"
echo "N GPUS: $n_gpus"
echo "GPU Type: $gpu_type"
echo "Queue: $queue"
echo "Memory: $mem GB"
echo "Comment: $comment"


# prepare the environment and load necessary modules
cd /mnt/users/jmalone/GitHub/yalla/run
rm exec*
rm output/*
module load cuda

# generate .h file from default parameters
source ../venv/bin/activate
if [[ "$version" -eq "0" ]]; then
    python3 ../sample/pwriter.py
elif [[ "$version" -eq "1" ]]; then
    python3 ../sample/pwriter.py -p volk_params.csv
elif [[ "$version" -eq "2" ]]; then
    python3 ../sample/pwriter.py -p eggspot_layers_params.csv
fi
deactivate

# compile the model
echo "Compiling..."
# nvcc -std=c++14 -arch=sm_61 ../examples/eggspot.cu -o exec
if [[ "$version" -eq "0" ]]; then
    /usr/local/cuda/bin/nvcc -std=c++17 -arch=sm_61 ../examples/eggspot.cu -o exec
elif [[ "$version" -eq "1" ]]; then
    /usr/local/cuda/bin/nvcc -std=c++17 -arch=sm_61 ../examples/volk.cu -o exec
elif [[ "$version" -eq "2" ]]; then
    /usr/local/cuda/bin/nvcc -std=c++17 -arch=sm_61 ../examples/eggspot_layers.cu -o exec
fi
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
if [[ "$version" -eq "0" ]]; then
    cp ../examples/eggspot.cu output/eggspot.cu # copy source code into output directory
    cp ../params/default.csv output/default.csv # copy params.h into output directory
elif [[ "$version" -eq "1" ]]; then
    cp ../examples/volk.cu output/volk.cu
    cp ../params/volk_params.csv output/volk_params.csv
elif [[ "$version" -eq "2" ]]; then
    cp ../examples/eggspot_layers.cu output/eggspot_layers.cu
    cp ../params/eggspot_layers_params.csv output/eggspot_layers_params.csv
fi