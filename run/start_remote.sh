#!/bin/bash
set -e # halt if error
ssh jmalone@hydra.physics.ox.ac.uk 'source /etc/profile; /mnt/users/jmalone/GitHub/yalla/run/start_model.sh'
rm -r /home/jmalone/GitHub/yalla/run/saves/output
scp -rpv jmalone@hydra.physics.ox.ac.uk:/mnt/users/jmalone/GitHub/yalla/run/output /home/jmalone/GitHub/yalla/run/saves
python3 /home/jmalone/GitHub/yalla/run/render.py output -c u