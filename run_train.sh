#!/bin/bash
# Tutorial:  https://linuxconfig.org/how-to-use-a-bash-script-to-run-your-python-scripts

# cd /home/Boudreault/Documents/Mandible_tracking ||

# Activate the conda env
source /home/Boudreault/anaconda3/etc/profile.d/conda.sh
conda activate pytorch-env

# echo $(pwd)

# Run train script
PYTHONPATH=/home/Boudreault/Documents/Mandible_tracking python siamese_net/train.py siamese_net/config.yaml