#!/bin/bash
# Tutorial:  https://linuxconfig.org/how-to-use-a-bash-script-to-run-your-python-scripts

# cd /path/to/python/script/directory ||

# Looping on multiple config files
echo 'Running script loading config using for loop'
for ID in 1 2 3
do
    echo "Training using' siamese_net/config$ID.yaml"
    python siamese_net/train.py siamese_net/config$ID.yaml
done