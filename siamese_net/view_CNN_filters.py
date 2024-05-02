# Taken from: https://gist.github.com/wangg12/f11258583ffcc4728eb71adc0f38e832#file-viz_net_pytorch-py
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

import utils
from SiameseNet import SiameseNetwork


def kernel_vis(tensor, ch=0, allkernels=False, nrow=8, padding=1):
    # Taken from https://discuss.pytorch.org/t/how-to-visualize-the-actual-convolution-filters-in-cnn/13850/8
    n, c, w, h = tensor.shape

    if allkernels:
        tensor = tensor.view(n * c, -1, w, h)
    elif c != 3:
        tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))
    grid = torchvision.utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)

    return grid.numpy().transpose((1, 2, 0)), nrow, rows


if __name__ == "__main__":
    config_file = pathlib.Path("config_windows.yaml")
    configs = utils.load_configs(config_file)
    subnet_name = configs['training']['sub_model']
    cam_inputs = configs['training']['cam_inputs']
    weights_file_addon = configs['training']['weights_file_addon']

    cam_str = ''.join([c[0].lower() for c in cam_inputs])
    if weights_file_addon:
        weights_file = f"{subnet_name}_{cam_str}cams_{configs['training']['num_fc_hidden_units']}_{weights_file_addon}"
    else:
        weights_file = f"{subnet_name}_{cam_str}cams_{configs['training']['num_fc_hidden_units']}_fully_trained"
    print(f'Loading weights from: {weights_file}')

    # Define the model
    model = SiameseNetwork(configs)
    model.load_state_dict(torch.load(f"model_weights/{weights_file}.pth"))

    # Get first conv layer weights
    subnet_layers = model.subnet
    first_conv = subnet_layers[0]
    print(f'First conv layer: {first_conv}')
    first_conv_filters = model.subnet[0].weight.clone()     # Shape: [# filters, # input channels, k_h, k_w]
    [first_feature_grid, nrow, rows] = kernel_vis(first_conv_filters, ch=0, allkernels=False)
    plt.figure(figsize=(nrow, rows))
    plt.imshow(first_feature_grid)
    plt.axis('off')
    plt.ioff()
    plt.title(f'Kernel values of the first {model.subnet_name} conv layer')
    plt.show(block=True)

    # Compare those values to the default ResNet-18 values
    """
    subnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    [default_feature_grid, nrow, rows] = kernel_vis(subnet.conv1.weight.clone(), ch=0, allkernels=False)
    plt.figure(figsize=(nrow, rows))
    plt.imshow(default_feature_grid)
    plt.axis('off')
    plt.ioff()
    plt.title(f'Default kernel values of the first {model.subnet_name} conv layer')
    plt.show(block=True)
    """

    # Plot last filter values
    """
    last_conv = subnet_layers[-1][-1].conv2
    last_conv_filters = last_conv.weight.clone()
    [last_feature_grid, nrow, rows] = kernel_vis(last_conv_filters, ch=0, allkernels=True, nrow=512)
    plt.figure(figsize=(nrow, rows))
    plt.imshow(last_feature_grid)
    plt.axis('off')
    plt.ioff()
    plt.title(f'Kernel values of the last {model.subnet_name} conv layer')
    plt.show(block=True)
    """

    # TODO: Get activations after each block to see when the screws start to be recognized
    dataset_root = pathlib.Path(configs['data']['dataset_root'])
    sample_img_l = dataset_root.joinpath(r'Left/traj_contour0_0_l.jpg')
    sample_img_r = dataset_root.joinpath(r'Right/traj_contour0_0_r.jpg')
    sample_img_s = dataset_root.joinpath(r'Side/traj_contour0_0_s.jpg')


