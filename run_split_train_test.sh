#!/bin/bash
# Tutorial:  https://linuxconfig.org/how-to-use-a-bash-script-to-run-your-python-scripts

# cd /home/Boudreault/Documents/Mandible_tracking ||

# Activate the conda env
source /home/Boudreault/anaconda3/etc/profile.d/conda.sh
conda activate pytorch-env

# echo $(pwd)

# Run the split_train_test script
PYTHONPATH=/home/Boudreault/Dokumente/Mandible_tracking python data_prep/split_train_test data_prep/data_config.yaml
