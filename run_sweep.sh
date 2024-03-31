#!/bin/bash
# Tutorial:  https://linuxconfig.org/how-to-use-a-bash-script-to-run-your-python-scripts

# cd /path/to/python/script/directory ||

# Activate the conda env
source /home/Boudreault/anaconda3/etc/profile.d/conda.sh
conda activate pytorch-env

# Looping on multiple config files
echo 'Running script loading config using for loop'
for ID in 1 2 3 4 5 6 7 8
do
    echo "Training using' siamese_net/config$ID.yaml"
    PYTHONPATH=/home/Boudreault/Documents/Mandible_tracking python siamese_net/train.py siamese_net/sweep_configs/config$ID.yaml
    # python main.py siamese_net/sweep_configs/config$ID.yaml
done