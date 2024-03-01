import pathlib
import numpy as np
import matplotlib as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torch.utils.data import DataLoader

# Disable warning for using transforms.v2
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2

import utils
import utils_data
from utils_data import MandibleDataset, NormTransform
from SiameseNet import SiameseNetwork, get_preds


if __name__ == '__main__':
    # Load configs
    config_file = pathlib.Path("siamese_net/config.yaml")
    configs = utils.load_configs(config_file)
    dataset_root = pathlib.Path(configs['data']['dataset_root'])
    anno_paths_test = configs['data']['trajectories_test']

    subnet_name = configs['training']['sub_model']
    cam_inputs = configs['training']['cam_inputs']
    test_bs = configs['training']['test_bs']

    # Set training device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Inference done on {device}')

    # Merge all annotation files based on config file
    print("Loading annotation files...")
    annotations_test = utils_data.merge_annotations(dataset_root, anno_paths_test)
    annotations_test = annotations_test[:5]

    # Create dataset object
    print("Initializing dataset object...")
    # Create dataset objects
    transforms = v2.Compose([NormTransform()])  # Remember to also change the annotations for other transforms
    dataset_test = MandibleDataset(dataset_root, cam_inputs, annotations_test, transforms)

    print("Creating dataloader...")
    dataloader_test = DataLoader(dataset_test, batch_size=test_bs, shuffle=False, num_workers=4)

    # Define the model
    model = SiameseNetwork(configs)
    # Load trained weights
    weights_file = f"{subnet_name}_{len(cam_inputs)}cams_{configs['training']['num_fc_hidden_units']}"
    model.load_state_dict(torch.load(f"siamese_net/model_weights/{weights_file}.pth"))
    model.to(device)

    print("Performing inference...")
    preds = get_preds(model, device, dataloader_test)

    # Save preds as csv
    print("Saving results...")
    preds_file = f"{subnet_name}_{len(cam_inputs)}cams_{configs['training']['num_fc_hidden_units']}"
    preds.to_csv(preds_file)

