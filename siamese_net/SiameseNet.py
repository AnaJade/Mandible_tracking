"""
Code to build a siamese network using 3 sub-networks and output a 7D [pos_x, pos_y, pos_z, q1, q2, q3, q4] tensor
Based on the PyTorch example: https://github.com/pytorch/examples/blob/main/siamese_network/main.py
"""
import math
import pathlib
import copy
import time

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.interpolate import griddata

import wandb
import torch
import torch.nn as nn
import torchvision
from torchvision import models

import utils
import utils_data
from EfficientNet import EfficientNet


class SiameseNetwork(nn.Module):
    def __init__(self, configs: dict, input_shape=(1, 3, 1200, 1920)):
        super(SiameseNetwork, self).__init__()
        self.input_shape = input_shape
        self.subnet_name = configs['training']['sub_model']
        self.use_pretrained = configs['training']['use_pretrained']
        self.cam_inputs = configs['training']['cam_inputs']
        self.num_subnets = len(self.cam_inputs)
        self.nb_hidden = configs['training']['num_fc_hidden_units']

        # Init model variables
        self.subnet = None
        self.subnet_output_units = 0

        # Init average pool layer that needs to be manually added to the EfficientNet feature extractor
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

        # Get subnet network
        if self.subnet_name == 'resnet18':
            self.resnet18_init()
        elif "efficientnet" in self.subnet_name:
            self.efficientnet_init()
        else:
            print(f"Desired subnetwork ({self.subnet_name}) isn't defined.")

        # Add linear layers to compare between the features of the images
        self.fc = nn.Sequential(
            nn.Linear(self.subnet_output_units * self.num_subnets, self.nb_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.nb_hidden, 7),  # Output set to 7 for 3 position and 4 orientation variables
        )
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def resnet18_init(self):
        # get resnet model
        if self.use_pretrained:
            self.subnet = torchvision.models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            self.subnet = torchvision.models.resnet18(weights=None)

        self.subnet_output_units = self.subnet.fc.in_features

        # remove the last 2 layers of resnet18 (avg pool and linear layer) to keep only the feature extraction layers
        self.subnet = torch.nn.Sequential(*(list(self.subnet.children())[:-2]))

        # Load weights if needed
        if not self.use_pretrained:
            self.subnet.apply(self.init_weights)

    def efficientnet_init(self):
        # Check if the version of efficientnet is valid
        if self.subnet_name in [f'efficientnet-b{v}' for v in range(8)]:
            # Init the entire EfficientNet model
            if self.use_pretrained:
                self.subnet = EfficientNet.from_pretrained(self.subnet_name)
            else:
                self.subnet = EfficientNet.from_name(self.subnet_name)
            # Get subnet output size
            # TODO: Find a better way to calculate the nb of output units
            dummy_input = torch.randn(self.input_shape)  # Input shape of the images
            subnet_output = self.forward_subnet(dummy_input)
            self.subnet_output_units = subnet_output.shape[-1]
        else:
            print(f"{self.subnet_name} isn't a valid EfficientNet version")
            # Available versions: 'efficientnet-b{i}', with i = 0, ..., 7
            raise SystemExit(1)

    def extract_features(self, x):
        # x shape: [batch, 3, 1200, 1920], dtype=torch.float32
        if 'resnet' in self.subnet_name:
            output = self.subnet(x)
        elif 'efficientnet' in self.subnet_name:
            output = self.subnet.extract_features(x)
        else:
            output = torch.Tensor()
        return output

    def extract_features_subnets(self, imgs):
        # Image feature extraction
        outputs = [self.extract_features(img) for img in imgs]
        return outputs

    def forward_subnet(self, x):
        # x shape: [batch, 3, 1200, 1920], dtype=torch.float32
        if 'resnet' in self.subnet_name:
            output = self.extract_features(x)
            output = self.avg_pooling(output)  # Shape: [batch, # feature maps, 1, 1]
            output = output.view(output.size()[0], -1)  # Shape: [batch, # feature maps], dtype=torch.float32
        elif 'efficientnet' in self.subnet_name:
            output = self.subnet.extract_features(x)
            output = self.avg_pooling(output)
            output = output.view(output.size()[0], -1)  # Shape: [batch, # feature maps], dtype=torch.float32
        else:
            output = torch.Tensor()
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
    current_batch = 0

    # Version with raising another exception (stops when an empty file is found)
    dataloader_iterator = iter(tqdm(enumerate(train_loader)))
    while True:
        try:
            batch_idx, (images, targets) = next(dataloader_iterator)
            current_batch = batch_idx
            # Train as usual
            images = [img.to(device) for img in images]
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            # Reshape outputs/targets if needed
            if outputs.shape != targets.shape:
                outputs = outputs.reshape([-1, 7])
                targets = targets.reshape([-1, 7])
            batch_loss = criterion(outputs, targets)  # .type(torch.float32)
            batch_loss.backward()
            optimizer.step()

            # Log batch loss
            if log_wandb:
                wandb_log('batch', train_loss=batch_loss.item())

            # Update epoch loss
            epoch_loss = batch_loss.sum().item()

        except StopIteration:
            break
        except FileNotFoundError as err:
            print(err)

    # Calculate final epoch loss
    if current_batch > 0:
        epoch_loss /= (current_batch*train_loader.batch_size)
    else:
        print(f"Epoch loss couldn't be calculated because first loaded image couldn't be read")
        epoch_loss = None

    # Version with the warning (runs VERY SLOWLY)
    """
    for batch_idx, (images, targets) in tqdm(enumerate(train_loader)):
        # Check if any images in the batch couldn't be read
        #   Images: [cam batch_img_tensor for cam in cameras] -> batch_img_tensor.shape = [batch, 3, 1200, 1920]
        img_cam0 = images[0]
        img_cam0 = img_cam0.reshape([img_cam0.shape[0], -1])
        sum_img_cam0 = torch.sum(img_cam0, dim=1)
        usable_imgs = (sum_img_cam0 != 0).flatten()
        nb_invalid_imgs += (sum_img_cam0 == 0).sum()
        if not torch.any(usable_imgs):
            continue
        else:
            # Remove non-valid images
            images = [img[usable_imgs, ...] for img in images]
            targets = targets[usable_imgs, ...]
            # Train as usual
            images = [img.to(device) for img in images]
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            # Reshape outputs/targets if needed
            if outputs.shape != targets.shape:
                outputs = outputs.reshape([-1, 7])
                targets = targets.reshape([-1, 7])
            batch_loss = criterion(outputs, targets)  # .type(torch.float32)
            batch_loss.backward()
            optimizer.step()

            # Log batch loss
            if log_wandb:
                wandb_log('batch', train_loss=batch_loss.item())

            # Update epoch loss
            epoch_loss = batch_loss.sum().item()

    # Calculate final epoch loss
    epoch_loss /= (len(train_loader.dataset) - nb_invalid_imgs)
    """

    return epoch_loss, model


def eval_model(model, device, dataloader, criterion):
    model.eval()
    valid_loss = 0
    nb_invalid_imgs = 0
    current_batch = 0
    with torch.no_grad():
        # Version with raising another exception (stops when an empty file is found)
        dataloader_iterator = iter(tqdm(enumerate(dataloader)))
        valid_preds = []
        valid_targets = []
        while True:
            try:
                batch_idx, (images, targets) = next(dataloader_iterator)
                current_batch = batch_idx
                # Evaluate as usual
                images = [img.to(device) for img in images]
                targets = targets.to(device)
                outputs = model(images)  # .squeeze()
                # Reshape outputs/targets if needed
                if outputs.shape != targets.shape:
                    outputs = outputs.reshape([-1, 7])
                    targets = targets.reshape([-1, 7])
                    valid_preds.append(outputs)
                    valid_targets.append(targets)
                valid_loss += criterion(outputs, targets).sum().item()  # sum up batch loss
            except StopIteration:
                break
            except FileNotFoundError as err:
                print(err)

    # Calculate final epoch loss
    if current_batch > 0:
        valid_loss /= (current_batch * dataloader.batch_size)
    else:
        print(f"Valid loss couldn't be calculated because first loaded image couldn't be read")
        valid_loss = 1e7    # Return random large value

    # Calculate the loss per dimension
    # valid_preds = utils.pose_quaternion2euler(np.concatenate([p.detach().cpu() for p in valid_preds], axis=0))
    # valid_targets = utils.pose_quaternion2euler(np.concatenate([t.detach().cpu() for t in valid_targets], axis=0))
    valid_preds = np.concatenate([p.detach().cpu() for p in valid_preds], axis=0)
    valid_targets = np.concatenate([t.detach().cpu() for t in valid_targets], axis=0)
    valid_loss_per_dim = utils_data.get_loss_per_axis(valid_preds, valid_targets)

    # Version with the warning (runs VERY SLOWLY)
    """
        for (images, targets) in tqdm(dataloader):
            # Check if any images in the batch couldn't be read
            #   Images: [cam batch_img_tensor for cam in cameras] -> batch_img_tensor.shape = [batch, 3, 1200, 1920]
            img_cam0 = images[0]
            img_cam0 = img_cam0.reshape([img_cam0.shape[0], -1])
            sum_img_cam0 = torch.sum(img_cam0, dim=1)
            usable_imgs = (sum_img_cam0 != 0).flatten()
            nb_invalid_imgs += (sum_img_cam0 == 0).sum()
            if not torch.any(usable_imgs):
                continue
            else:
                # Remove non-valid images
                images = [img[usable_imgs, ...] for img in images]
                targets = targets[usable_imgs, ...]
                # Evaluate as usual
                images = [img.to(device) for img in images]
                targets = targets.to(device)
                outputs = model(images)  # .squeeze()
                # Reshape outputs/targets if needed
                if outputs.shape != targets.shape:
                    outputs = outputs.reshape([-1, 7])
                    targets = targets.reshape([-1, 7])
                valid_loss += criterion(outputs, targets).sum().item()  # sum up batch loss

    valid_loss /= (len(dataloader.dataset) - nb_invalid_imgs)
    """

    return valid_loss, valid_loss_per_dim


def train_model(configs, model, dataloaders, device, criterion, optimizer, scheduler):
    # Load relevant config params
    log_wandb = configs['wandb']['wandb_log']
    num_epochs = configs['training']['max_epochs']
    patience = configs['training']['patience']

    subnet_name = configs['training']['sub_model']
    cam_inputs = configs['training']['cam_inputs']
    nb_hidden = configs['training']['num_fc_hidden_units']
    weights_file_addon = configs['training']['weights_file_addon']

    train_loader = dataloaders[0]
    valid_loader = dataloaders[1]

    cam_str = ''.join([c[0].lower() for c in cam_inputs])
    if weights_file_addon:
        weights_file = f"{subnet_name}_{cam_str}cams_{nb_hidden}_{weights_file_addon}"
    else:
        weights_file = f"{subnet_name}_{cam_str}cams_{nb_hidden}"

    if log_wandb:
        wandb_init(configs)

    # Train model
    best_valid_loss = 1e6  # Random large number
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
        epoch_valid_loss, epoch_valid_loss_per_dim = eval_model(model, device, valid_loader, criterion)
        scheduler.step()

        print(f'Epoch {epoch} results ----------------------')
        print(f'Train loss: {epoch_train_loss}')
        print(f'Valid loss: {epoch_valid_loss}')
        print(f'Valid loss per dim: \n{epoch_valid_loss_per_dim}')

        if epoch_valid_loss < best_valid_loss:
            best_epoch = epoch
            best_valid_loss = epoch_valid_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            # Save model
            torch.save(model.state_dict(), f"siamese_net/model_weights/{weights_file}.pth")

        # Log results
        if log_wandb:
            if len(epoch_valid_loss_per_dim) == 6:
                wandb_log('epoch',
                          train_loss=epoch_train_loss,
                          valid_loss=epoch_valid_loss,
                          valid_x_mse=epoch_valid_loss_per_dim.loc['x', 'MSE'],
                          valid_y_mse=epoch_valid_loss_per_dim.loc['y', 'MSE'],
                          valid_z_mse=epoch_valid_loss_per_dim.loc['z', 'MSE'],
                          valid_rx_mse=epoch_valid_loss_per_dim.loc['rx', 'MSE'],
                          valid_ry_mse=epoch_valid_loss_per_dim.loc['ry', 'MSE'],
                          valid_rz_mse=epoch_valid_loss_per_dim.loc['rz', 'MSE'])
            elif len(epoch_valid_loss_per_dim) == 7:
                wandb_log('epoch',
                          train_loss=epoch_train_loss,
                          valid_loss=epoch_valid_loss,
                          valid_x_mse=epoch_valid_loss_per_dim.loc['x', 'MSE'],
                          valid_y_mse=epoch_valid_loss_per_dim.loc['y', 'MSE'],
                          valid_z_mse=epoch_valid_loss_per_dim.loc['z', 'MSE'],
                          valid_q1_mse=epoch_valid_loss_per_dim.loc['q1', 'MSE'],
                          valid_q2_mse=epoch_valid_loss_per_dim.loc['q2', 'MSE'],
                          valid_q3_mse=epoch_valid_loss_per_dim.loc['q3', 'MSE'],
                          valid_q4_mse=epoch_valid_loss_per_dim.loc['q4', 'MSE'])

    if log_wandb:
        wandb.finish()

    model.load_state_dict(best_model_weights)

    return model


def get_preds(model, device, dataloader, min_max_pos=None) -> pd.DataFrame:
    """
    Get pose predictions on the dataloader data
    :param model: model with the pre-trained weights already loaded
    :param device: device used to perform inference
    :param dataloader: pytorch dataloader with the inference data
    :param min_max_pos: min and max possible values in x, y, z as a list: [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    :return: model predictions for the images in the dataloader
    """
    # Get predictions
    preds = []
    inference_time = []
    with torch.no_grad():
        for (images, targets) in tqdm(dataloader):
            images = [img.to(device) for img in images]
            start_time = time.time()
            output = model(images).squeeze()
            inference_time.append(time.time() - start_time)
            preds.append(output)
    # Merge all outputs into one array
    if len(preds[-1].shape) < len(preds[-2].shape):
        preds[-1] = preds[-1][None, :]
    preds = torch.concat(preds, dim=0).detach().cpu().numpy()
    if min_max_pos is not None:
        preds = utils_data.rescale_position(preds, np.array(min_max_pos[0]), np.array(min_max_pos[1]))
    # Format to df
    preds = pd.DataFrame(preds)
    # Find inference time stats
    inference_time = np.array(inference_time)
    print(f'Average inference time: {np.mean(inference_time)}')
    print(f'Min inference time: {np.min(inference_time)}')
    print(f'Max inference time: {np.max(inference_time)}')

    return preds


def overlay_activation_map(imgs: list[np.ndarray], heatmaps: list[np.ndarray]) -> list:
    """
    Format the heatmaps and overlay it on the original image
    :param imgs: list with the original images
    :param heatmaps: list with the subnet outputs
    :return: list with the superimposed images
    """
    heatmaps = [np.maximum(np.mean(heatmap, axis=0), 0)/np.max(heatmap) for heatmap in heatmaps]
    heatmaps = [cv2.resize(heatmap, (imgs[0].shape[1], imgs[0].shape[0])) for heatmap in heatmaps]
    heatmaps = [np.uint8(255 * (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))) for heatmap in heatmaps]
    heatmaps = [cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) for heatmap in heatmaps]
    overlayed_imgs = [cv2.addWeighted(heatmap, 0.35, img, 0.65, 0.0) for heatmap, img in zip(heatmaps, imgs)]
    return overlayed_imgs


def get_feature_maps(model: SiameseNetwork, device, dataloader, cam_inputs):
    """
    Get the saliency maps for images in the given dataloader
    :param model: Siamese model with the weights preloaded
    :param device: device used to get feature maps
    :param dataloader: dataloader with the images
    :param cam_inputs: list with the cameras being used
    :return:
    """
    with torch.no_grad():
        for i, (images, targets) in tqdm(enumerate(dataloader)):
            images = [img.to(device) for img in images]
            outputs = model.extract_features_subnets(images)
            # Reformat images
            images = [img[0, ...].detach().cpu().numpy() for img in images]
            outputs = [output[0, ...].detach().cpu().numpy() for output in outputs]
            images = [(img.transpose((1, 2, 0)) * 255).astype('uint8') for img in images]
            # Overlay feature maps
            heatmap_imgs = overlay_activation_map(images, outputs)
            # Display results
            heatmaps = [np.maximum(np.mean(heatmap, axis=0), 0) / np.max(heatmap) for heatmap in outputs]
            plt.figure(figsize=(3 * len(heatmap_imgs), 7))
            img_name = dataloader.dataset.img_names[i]
            for j, (heatmap_img, heatmap, img, cam) in enumerate(zip(heatmap_imgs, heatmaps, images, cam_inputs)):
                plt.subplot(3, len(heatmap_imgs), j+1)
                plt.title(cam)
                plt.imshow(heatmap_img)
                plt.axis('off')
                plt.subplot(3, len(heatmap_imgs), j+4)
                if j == 0:
                    plt.title('Subnet outputs:', loc='left')
                plt.imshow(heatmap)
                # plt.axis('off')
                plt.subplot(3, len(heatmap_imgs), j+7)
                if j == 0:
                    plt.title('Input images:', loc='left')
                plt.imshow(img)
                # plt.axis('off')
            plt.suptitle(f'{img_name} saliency map')
            plt.tight_layout()
            plt.show(block=True)


def get_plane_error(model: SiameseNetwork, device, dataloader, min_max_pos, plane, grid_size):
    """
    Get the average error of every image per "bin" in the given plane
    :param model: Siamese model with the weights preloaded
    :param device: device used to get feature maps
    :param dataloader: dataloader (pre-filtered) with the images
    :param min_max_pos: min and max possible values in x, y, z as a list: [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    :param plane: string that is either 'xy', 'xz', or 'yz'
    :param grid_size: number of sections in which the plane's 2 axis will be split
    :return:
    """
    # Parse plane parameter
    plane = [*plane]
    rmse_axis = [ax for ax in ['x', 'y', 'z'] if ax not in plane][0]
    plane_axis_id = [i for i, ax in enumerate(['x', 'y', 'z']) if ax in plane]
    rmse_axis_id = [i for i, ax in enumerate(['x', 'y', 'z']) if ax not in plane][0]
    # Get the MSE in the other axis
    try:
        plane_results = pd.read_csv(f"temp_plane{"".join(plane)}_results.csv")  # Remove after debug
    except FileNotFoundError:
        plane_results = []
        with torch.no_grad():
            for (images, targets) in tqdm(dataloader):
                images = [img.to(device) for img in images]
                output = model(images).squeeze()
                # Rescale to true position
                if min_max_pos is not None:
                    preds = utils_data.rescale_position(output[None, :].detach().cpu().numpy(),
                                                        np.array(min_max_pos[0]), np.array(min_max_pos[1]))
                    targets = utils_data.rescale_position(targets[0, ...].detach().cpu().numpy(),
                                                          np.array(min_max_pos[0]), np.array(min_max_pos[1]))
                # Get MSE on remaining axis
                rmse = mean_squared_error(targets[:, rmse_axis_id], preds[:, rmse_axis_id], squared=False)
                plane_results.append({plane[0]: targets[0, plane_axis_id[0]],
                                      plane[1]: targets[0, plane_axis_id[1]],
                                      rmse_axis: targets[0, rmse_axis_id],
                                      f'rmse_{rmse_axis}': rmse})
        plane_results = pd.DataFrame(plane_results)
        plane_results.to_csv(f"temp_plane{"".join(plane)}_results.csv", index=False)     # Remove after debug

    # Create grid limits
    min_max_ax0 = [math.floor(plane_results[plane[0]].min()), math.ceil(plane_results[plane[0]].max())]
    min_max_ax1 = [math.floor(plane_results[plane[1]].min()), math.ceil(plane_results[plane[1]].max())]
    bins_ax0 = list(np.ceil(np.linspace(start=min_max_ax0[0], stop=min_max_ax0[1], num=grid_size+1)).astype(np.int32))
    bins_ax1 = list(np.ceil(np.linspace(start=min_max_ax1[0], stop=min_max_ax1[1], num=grid_size+1)).astype(np.int32))
    bin_centers_ax0 = [(bins_ax0[i] + bins_ax0[i+1])/2 for i in range(grid_size)]
    bin_centers_ax1 = [(bins_ax1[i] + bins_ax1[i+1])/2 for i in range(grid_size)]
    bin_centers_ax1.reverse()   # Needed to have the vertical axis thr right way on the graph

    # Split the images into the grid
    error_grid = pd.DataFrame(index=bin_centers_ax1, columns=bin_centers_ax0)
    img_count_grid = pd.DataFrame(index=bin_centers_ax1, columns=bin_centers_ax0)
    for i, bin0_id in enumerate(bin_centers_ax0):
        # Filter to keep only relevant images based on ax 0
        bin_ax0_results = plane_results[plane_results[plane[0]].between(bins_ax0[i], bins_ax0[i+1])]
        for j, bin1_id in enumerate(bin_centers_ax1):
            # Filter to keep only relevant images based on ax 1
            bin_results = bin_ax0_results[bin_ax0_results[plane[1]].between(bins_ax1[j], bins_ax1[j+1])]
            # Calculate average error
            error_grid.loc[bin1_id, bin0_id] = bin_results[f'mse_{rmse_axis}'].mean()
            # Save image count per grid cell
            img_count_grid.loc[bin1_id, bin0_id] = len(bin_results)
    # Fill NaN values with numpy nan (needed to plot the results)
    error_grid.fillna(np.nan, inplace=True)
    print(error_grid)
    print(img_count_grid)

    # Plot and show results
    grid_data = error_grid.to_numpy().astype(float)
    img_count_data = img_count_grid.to_numpy()
    cmap = matplotlib.cm.get_cmap('viridis')
    cmap.set_bad(color='white')
    plt.imshow(grid_data, extent=tuple([float(i) for i in min_max_ax0 + min_max_ax1]), cmap=cmap)
    # Overlay values
    for (i, j), rmse in np.ndenumerate(grid_data):
        plt.text(bin_centers_ax0[j], bin_centers_ax1[i], f'{round(rmse, 2)}\n{img_count_data[i, j]} imgs',
                 bbox=dict(facecolor='white', alpha=0.5),
                 ha='center', va='center')
    plt.colorbar()
    plt.xlabel(plane[0])
    plt.ylabel(plane[1])
    plt.title(f'Average {rmse_axis.upper()} position RMSE in the {"".join(plane).upper()} plane')
    plt.figtext(.5, .05, 'White sections represent NaN values due to a lack of images', ha='center')
    plt.tight_layout()
    plt.show(block=True)


def wandb_init(configs: dict):
    wandb.init(
        # Choose wandb project
        project=configs['wandb']['project_name'],
        # Add hyperparameter tracking
        config={
            "trajectories_train": configs['data']['trajectories_train'],
            "trajectories_valid": configs['data']['trajectories_valid'],
            "rescale_pos": configs['data']['rescale_pos'],
            "subnet": configs['training']['sub_model'],
            "weights_file_addon": configs['training']['weights_file_addon'],
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
    # print(model.print_layers())

    img = torch.randn(1, 3, 1200, 1920)

    subnet_output = model.forward_subnet(img)

    print(f'Shape of subnet output: {subnet_output.shape}')
