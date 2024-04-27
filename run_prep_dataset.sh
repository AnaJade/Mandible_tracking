#!/bin/bash
# Tutorial:  https://linuxconfig.org/how-to-use-a-bash-script-to-run-your-python-scripts

# cd /home/Boudreault/Documents/Mandible_tracking ||

# Activate the conda env
source /home/Boudreault/anaconda3/etc/profile.d/conda.sh
conda activate pytorch-env

# echo $(pwd)

# Run the get_pose_for_frames script
PYTHONPATH=/home/Boudreault/Dokumente/Mandible_tracking python data_prep/get_pose_for_frames data_prep/data_config.yaml

# Run the prep_dataset script
PYTHONPATH=/home/Boudreault/Dokumente/Mandible_tracking python data_prep/prep_dataset data_prep/data_config.yaml