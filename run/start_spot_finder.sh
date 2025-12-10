#!/bin/bash
# This script is used to compile and run the model on the cluster

# cluster parameters
n_gpus=1
gpu_type="rtx2080with12gb"  # rtx2080with12gb, rtx3090with24gb
queue="gpulong"            # gpushort, gpulong
mem=4                       # allocated memory in GB
comment="none"            # comment for the job
out_dir_name="sample_test"      # default output directory name

# get command line arguments
POSITIONAL_ARGS=()      # array to hold positional arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -o|--outdir)
        out_dir_name="$2"
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

mkdir -p ${out_dir_name}

# prepare the environment and load necessary modules
source /etc/profile
cd /mnt/users/jmalone/GitHub/yalla/run
rm -f exec*
rm -rf ${out_dir_name}/*
module load cuda
module load python/3.11.4

# generate .h file from default parameters
source ../venv/bin/activate
python3 ../sample/pwriter.py -p eggspot_layers_params.csv
deactivate

# compile the model
echo "Compiling..."
source /etc/profile
# nvcc -std=c++14 -arch=sm_61 ../sample/spot_finder.cu -o exec
# echo "Compilation time: $SECONDS seconds"

PYTHON_INCLUDE="-I/usr/local/shared/python/3.11.4/include/python3.11"
PYTHON_LIBDIR="-L/usr/local/shared/python/3.11.4/lib"
PYTHON_LIBS="-lpython3.11 -lm -ldl"

nvcc -std=c++14 -arch=sm_61 \
  ${PYTHON_INCLUDE} \
  ../sample/spot_finder.cu \
  ${PYTHON_LIBDIR} \
  -Xlinker -rpath=/usr/local/shared/python/3.11.4/lib \
  ${PYTHON_LIBS} \
  -o exec

echo "Compilation time: $SECONDS seconds"

# remove queue lock file if it exists
if [ -f /mnt/users/jmalone/.addqueuelock ]; then
    addqueue -U
fi

# execute compiled model on cluster and capture console output
addqueue -c $comment -q $queue -s --gpus $n_gpus --gputype $gpu_type -m $mem ./exec ${out_dir_name}

cp ../examples/eggspot_layers.cu ${out_dir_name}/ # copy source code into output directory
cp ../sample/spot_finder.cu ${out_dir_name}/ # copy source code into output directory
