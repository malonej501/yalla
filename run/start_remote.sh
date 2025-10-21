#!/bin/bash
# set -e # continue if error

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

ssh jmalone@hydra.physics.ox.ac.uk "source /etc/profile; /mnt/users/jmalone/GitHub/yalla/run/start_model.sh -v ${version}"
if [ -d "/home/jmalone/GitHub/yalla/run/saves/output" ]; then
    rm -r /home/jmalone/GitHub/yalla/run/saves/output
fi
scp -rpv jmalone@hydra.physics.ox.ac.uk:/mnt/users/jmalone/GitHub/yalla/run/output /home/jmalone/GitHub/yalla/run/saves
python3 /home/jmalone/GitHub/yalla/run/render.py saves/output