#!/bin/bash
# Tutorial:  https://linuxconfig.org/how-to-use-a-bash-script-to-run-your-python-scripts

# cd /home/Boudreault/Documents/Mandible_tracking ||

# Activate the conda env
source /home/Boudreault/anaconda3/etc/profile.d/conda.sh
conda activate pytorch-env

# Run train script
PYTHONPATH=/home/Boudreault/Dokumente/Mandible_tracking python siamese_net/view_CNN_filters.py siamese_net/config.yaml