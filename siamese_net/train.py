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
from SiameseNet import SiameseNetwork, train_model

if __name__ == '__main__':
    # TODO: Change loss to account for the position and orientation scale variation
    # Load configs
    config_file = pathlib.Path("siamese_net/config.yaml")
    configs = utils.load_configs(config_file)
    dataset_root = pathlib.Path(configs['data']['dataset_root'])
    anno_paths_train = configs['data']['trajectories_train']
    anno_paths_valid = configs['data']['trajectories_valid']

    subnet_name = configs['training']['sub_model']
    cam_inputs = configs['training']['cam_inputs']
    train_bs = configs['training']['train_bs']
    valid_bs = configs['training']['valid_bs']

    nb_epochs = configs['training']['max_epochs']
    patience = configs['training']['patience']
    lr = configs['training']['lr']
    scheduler_step_size = configs['training']['lr_scheduler']['step_size']
    scheduler_gamma = configs['training']['lr_scheduler']['gamma']

    wandb_log = configs['wandb']['wandb_log']
    project_name = configs['wandb']['project_name']

    # Set training device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Training done on {device}')

    # Merge all annotation files based on config file
    print("Loading annotation files...")
    annotations_train = utils_data.merge_annotations(dataset_root, anno_paths_train)
    annotations_valid = utils_data.merge_annotations(dataset_root, anno_paths_valid)

    # Create dataset object
    print("Initializing dataset object...")
    # Create dataset objects
    transforms = v2.Compose([NormTransform()])  # Remember to also change the annotations for other transforms
    dataset_train = MandibleDataset(dataset_root, cam_inputs, annotations_train, transforms)
    dataset_valid = MandibleDataset(dataset_root, cam_inputs, annotations_valid, transforms)

    print("Creating dataloader...")
    dataloader_train = DataLoader(dataset_train, batch_size=train_bs, shuffle=True, num_workers=4)
    dataloader_valid = DataLoader(dataset_valid, batch_size=valid_bs, shuffle=False, num_workers=4)

    # Define the model
    model = SiameseNetwork(configs)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    # Define the criterion, optimizer and scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(params, lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    print("Training model...")
    print(f"Logging on wandb: {wandb_log}")
    model_fit = train_model(configs, model, [dataloader_train, dataloader_valid],
                            device, criterion, optimizer, scheduler)

    # Save model
    print("Saving best model...")
    weights_file = f"{subnet_name}_{len(cam_inputs)}cams_{configs['training']['num_fc_hidden_units']}"
    torch.save(model_fit.state_dict(), f"siamese_net/model_weights/{weights_file}.pth")





