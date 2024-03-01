"""
Code to build a siamese network using 3 sub-networks and output a 7D [pos_x, pos_y, pos_z, q1, q2, q3, q4] tensor
Based on the PyTorch example: https://github.com/pytorch/examples/blob/main/siamese_network/main.py
"""
import pathlib
import copy

import pandas as pd
from tqdm import tqdm

import wandb
import torch
import torch.nn as nn
import torchvision
from torchvision import models

import utils


class SiameseNetwork(nn.Module):
    def __init__(self, configs: dict):
        super(SiameseNetwork, self).__init__()
        self.subnet_name = configs['training']['sub_model']
        self.use_pretrained = configs['training']['use_pretrained']
        self.cam_inputs = configs['training']['cam_inputs']
        self.num_subnets = len(self.cam_inputs)
        self.nb_hidden = configs['training']['num_fc_hidden_units']

        # Init model variables
        self.subnet = None
        self.subnet_output_units = 0

        # Get subnet network
        if self.subnet_name == 'resnet18':
            self.resnet18_init()
        elif self.subnet_name == 'yolo6d':
            self.yolo6d_init()
        else:
            print(f"Desired subnetwork ({self.subnet_name}) isn't defined.")

        # Add linear layers to compare between the features of the images
        self.fc = nn.Sequential(
            nn.Linear(self.subnet_output_units * self.num_subnets, self.nb_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.nb_hidden, 7),   # Output set to 7 for 3 position and 4 orientation variables
        )
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def resnet18_init(self):
        # get resnet model
        if self.use_pretrained:
            self.subnet = torchvision.models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT)
        else:
            self.subnet = torchvision.models.resnet18(weights=None)

        self.subnet_output_units = self.subnet.fc.in_features

        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.subnet = torch.nn.Sequential(*(list(self.subnet.children())[:-1]))

        # Load weights if needed
        self.subnet.apply(self.init_weights)

    def yolo6d_init(self):
        pass

    def forward_subnet(self, x):
        output = self.subnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, inputs):
        # Image feature extraction
        output_subnet = [self.forward_subnet(img) for img in inputs]

        # concatenate both images' features
        output_subnets = torch.cat(output_subnet, 1)

        # pass the concatenation to the linear layers
        output = self.fc(output_subnets)

        return output.double()

    def print_layers(self):
        print(f"Subnet layers, repeated {self.num_subnets} times:")
        print(self.subnet)
        print("=========================================================")
        print("Fully connected layer")
        print(self.fc)


def train_1_epoch(model, device, train_loader, criterion, optimizer, log_wandb=False):
    model.train()
    epoch_loss = 0
    for batch_idx, (images, targets) in tqdm(enumerate(train_loader)):
        images = [img.to(device) for img in images]
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        batch_loss = criterion(outputs, targets).type(torch.float32)
        batch_loss.backward()
        optimizer.step()

        # Log batch loss
        if log_wandb:
            wandb_log('batch', train_loss=batch_loss.item())

        # Update epoch loss
        epoch_loss = batch_loss.sum().item()

        # For debugging purposes, save model
        if batch_idx == 40:
            weights_file = f"resnet18_2cams_256"
            torch.save(model.state_dict(), f"siamese_net/model_weights/{weights_file}.pth")

    # Calculate final epoch loss
    epoch_loss /= len(train_loader.dataset)

    return epoch_loss, model


def eval_model(model, device, dataloader, criterion):
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for (images, targets) in tqdm(dataloader):
            images = [img.to(device) for img in images]
            targets = targets.to(device)
            outputs = model(images).squeeze()
            valid_loss += criterion(outputs, targets).sum().item()  # sum up batch loss

    valid_loss /= len(dataloader.dataset)

    return valid_loss


def get_preds(model, device, dataloader) -> pd.DataFrame:
    # Get predictions
    preds = []
    with torch.no_grad():
        for (images, targets) in tqdm(dataloader):
            images = [img.to(device) for img in images]
            output = model(images).squeeze()
            print(f'Output: {output}')
            print(f'Output type: {type(output)}')
            print(f'Targets: {targets}')
            preds.append(output)
    # Merge all outputs into one array
    preds = torch.concat(preds, dim=0).detach().to_numpy()
    print(preds.shape)
    # Format to df
    preds = pd.DataFrame(preds)

    return preds


def train_model(configs, model, dataloaders, device, criterion, optimizer, scheduler):
    # Load relevant config params
    log_wandb = configs['wandb']['wandb_log']
    num_epochs = configs['training']['max_epochs']
    patience = configs['training']['patience']

    train_loader = dataloaders[0]
    valid_loader = dataloaders[1]

    if log_wandb:
        wandb_init(configs)

    # Train model
    best_valid_loss = 1e6     # Random large number
    best_epoch = 0
    best_model_weights = copy.deepcopy(model.state_dict())
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}  -----------------------------')
        if (epoch - best_epoch) >= patience:
            print(f'Loss has not improved for {patience} epochs. Training has stopped')
            print(f'Best loss was {best_valid_loss} @ epoch {best_epoch}')
            break
        epoch_train_loss, model = train_1_epoch(model, device, train_loader, criterion, optimizer, log_wandb)
        print('Calculating the loss on the valid set')
        epoch_valid_loss = eval_model(model, device, valid_loader, criterion)
        scheduler.step()

        print(f'Epoch {epoch} results ----------------------')
        print(f'Train loss: {epoch_train_loss}')
        print(f'Valid loss: {epoch_valid_loss}')

        if epoch_valid_loss < best_valid_loss:
            best_epoch = epoch
            best_model_weights = copy.deepcopy(model.state_dict())

        # Log results
        if log_wandb:
            wandb_log('epoch',
                      train_loss=epoch_train_loss,
                      valid_loss=epoch_valid_loss)

    if log_wandb:
        wandb.finish()

    model.load_state_dict(best_model_weights)

    return model


def wandb_init(configs: dict):
    wandb.init(
        # Choose wandb project
        project=configs['wandb']['project_name'],
        # Add hyperparameter tracking
        config={
            "trajectories_train": configs['data']['trajectories_train'],
            "trajectories_valid": configs['data']['trajectories_valid'],
            "subnet": configs['training']['sub_model'],
            "use_pretrained": configs['training']['use_pretrained'],
            "cam_inputs": configs['training']['cam_inputs'],
            "num_units_fc": configs['training']['num_fc_hidden_units'],
            "batch_size_train": configs['training']['train_bs'],
            "batch_size_valid": configs['training']['valid_bs'],
            "max_epochs": configs['training']['max_epochs'],
            "patience": configs['training']['patience'],
            "learning_rate": configs['training']['lr'],
            "scheduler_step_size": configs['training']['lr_scheduler']['step_size'],
            "scheduler_gamma": configs['training']['lr_scheduler']['gamma'],
        }
    )


def wandb_log(phase: str, **kwargs):
    """
    Log the given parameters
    :param phase: Either batch or epoch
    :param kwargs: Values to be logged
    :return:
    """
    # Append phase at the end of the param names
    log_data = {f'{key}_{phase}': value for key, value in kwargs.items()}
    wandb.log(log_data)


if __name__ == '__main__':
    config_file = pathlib.Path("siamese_net/config.yaml")
    configs = utils.load_configs(config_file)
    dataset_root = pathlib.Path(configs['data']['dataset_root'])
    anno_paths_train = configs['data']['trajectories_train']
    anno_paths_valid = configs['data']['trajectories_valid']

    cam_inputs = configs['training']['cam_inputs']
    train_bs = configs['training']['train_bs']
    valid_bs = configs['training']['valid_bs']

    # Set training device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Training done on {device}')

    # Define the model
    model = SiameseNetwork(configs)
    print(model.print_layers())

