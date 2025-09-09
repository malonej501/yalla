#!/bin/bash
# set -e # continue if error
ssh jmalone@hydra.physics.ox.ac.uk 'source /etc/profile; /mnt/users/jmalone/GitHub/yalla/run/start_model.sh'
if [ -d "/home/jmalone/GitHub/yalla/run/saves/output" ]; then
    rm -r /home/jmalone/GitHub/yalla/run/saves/output
fi
scp -rpv jmalone@hydra.physics.ox.ac.uk:/mnt/users/jmalone/GitHub/yalla/run/output /home/jmalone/GitHub/yalla/run/saves
python3 /home/jmalone/GitHub/yalla/run/render.py saves/output