#!/bin/bash
set -e # continue if error
ssh tplxdt135 'source /home/m/malone/.bashrc; /home/m/malone/GitHub/yalla/run/start_tplxdt135.sh'
if [ -d "/home/jmalone/GitHub/yalla/run/saves/output" ]; then
    rm -r /home/jmalone/GitHub/yalla/run/saves/output
fi
scp -rpv tplxdt135:/home/m/malone/GitHub/yalla/run/output /home/jmalone/GitHub/yalla/run/saves
python3 /home/jmalone/GitHub/yalla/run/render.py saves/output