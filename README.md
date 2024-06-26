# Mandible Tracking
Mandible pose tracking using a siamese network and RGB images.

## Packages and env setup
This code was tested using the following configuration: 
* PyTorch 2.2.2
* Cuda 11.8
* Nvidia driver 535

To install PyTorch, create a conda env and run `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`.
Also install the following packages: 
* matplotlib: `conda install matplotlib`
* opencv: `pip install opencv-python`
* pandas: `conda install pandas`
* sklearn: `conda install scikit-learn`
* tqdm: `conda install tqdm`
* wandb: `pip install wandb`

To log into wandb, run `wandb login [api key]`


## Config file parameters
The configuration file contains 3 main section: data, training and wandb.

**Data**
- `dataset_path`: Path to the root of the dataset
- `trajectories_train`: Annotation files for the trajectories to be used for training
- `trajectories_valid`: Annotation files for the trajectories to be used for the validation set
- `trajectories_test`: Annotation files for the trajectories to be used for the test set
- `rescale_pos`: Whether to scale the XYZ position values to between [-1, 1]

**Training**
- `sub_model`: Name of the sub-model that will be used in the model architecture
- `weights_file_addon`: String to be added at the end of the weight file name (can be used to create new file names during sweeps)
- `use_pretrained`: Whether to load the subnet pre-trained weights 
- `cam_imputs`: Images from these cameras will be used for training
- `num_fc_hidden_units`: Number of units in the fully connected hidden layer
- `train_bs`: Training batch size
- `valid_bs`: Validation batch size
- `test_bs`: Test batch size
- `max_epochs`: Maximum number of epochs that the model will be trained. This number can be large, since the training 
should stop on its own once performance on the valid set stops improving.
- `patience`: Number of epochs for which the model will continue training, even though no improvement is seen on the 
valid set performance.
- `lr`: Learning rate, impacts how much variation the model weights will have from one step to the other. This value 
determines the speed at which the model will learn. A value too small or too big can significantly increase the 
training time.
- `lr_scheduler`: A learning rate scheduler can be used to reduce the learning rate as training progresses. This means
that as the weights get closer to their local optimum, the step size for each update will be smaller to avoid 
overshooting.
  - `step_size`: Number of epochs between every learning rate update. The default value given in the tutorial is 3.
  - `gamma`: Value by which the learning rate will change. The default value given in the tutorial is 0.1.

**WandB**
- `wandb_log`: Choose if you want this experiment logged on weights and biases.
- `project_name`: Corresponds to the project name on wandb.

## Dataset file structure
Images need to be saved in a dataset folder containing the following sub-folders and files:
```
└── Dataset root
    └── Left
        ├── [traj name 0]_[frame no]_l.jpg
        ├── ...
        ├── [traj name 1]_[frame no]_l.jpg
        └── ...
    └── Right
        ├── [traj name 0]_[frame no]_r.jpg
        ├── ...
        ├── [traj name 1]_[frame no]_r.jpg
        └── ...
    └── Side
        ├── [traj name 0]_[frame no]_s.jpg
        ├── ...
        ├── [traj name 1]_[frame no]_s.jpg
        └── ...
    ├── [traj name 0].csv
    ├── [traj name 1].csv
    └── ...
```
The images are saved in the `Left`, `Right` and `Side` folders. 
The annotations are saved in `.csv` files using the following format:

| frame           | x   | y   | z   | q1  | q2   | q3  | q4  |
|-----------------|-----|-----|-----|-----|------|-----|-----|
| [traj name]_0   | 350 | 270 | 310 | 0.5 | -0.5 | 0.5 | 0.5 |
| [traj name]_1   | ... | ... | ... | ... | ...  | ... | ... |
| [traj name]_... | ... | ... | ... | ... | ...  | ... | ... |


## Data preprocessing
The XYZ position values will be scaled to fit between [-1, 1], which is also the range for the quaternion values.
Quaternions were kept for training, since the conversion to Euler angles isn't deterministic.

## Running the code on Ubuntu
**Bash file modifications**
1. Change the conda path and the env name
2. Change the `PYTHONPATH` value to the project folder path

**Running the bash scripts**
1. Open a terminal and go to the project folder
2. If necessary, change the bash file permissions with `chmod +x [bash file name].sh`
3. Run the script with `./[bash file name].sh`
