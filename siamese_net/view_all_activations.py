import argparse
import pathlib

from matplotlib import pyplot as plt
from tqdm import tqdm

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error

# Disable warning for using transforms.v2
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2

import utils
import utils_data
from utils_data import MandibleDataset, NormTransform
from SiameseNet import SiameseNetwork, get_feature_maps, overlay_activation_map

if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path',
                        help='Path to the config file',
                        type=str)

    args = parser.parse_args()
    config_file = pathlib.Path(args.config_path)    

    if not config_file.exists():
        print(f'Config file not found at {args.config_path}')
        raise SystemExit(1)

    # config_file = pathlib.Path("siamese_net/config_windows.yaml")

    # Load configs
    configs = utils.load_configs(config_file)
    dataset_root = pathlib.Path(configs['data']['dataset_root'])
    anno_paths_train = configs['data']['trajectories_train']
    anno_paths_valid = configs['data']['trajectories_valid']
    anno_paths_test = configs['data']['trajectories_test']
    resize_img_h = configs['data']['resize_img']['img_h']
    resize_img_w = configs['data']['resize_img']['img_w']
    grayscale = configs['data']['grayscale']
    rescale_pos = configs['data']['rescale_pos']

    subnet_name = configs['training']['sub_model']
    cam_inputs = configs['training']['cam_inputs']
    train_bs = configs['training']['train_bs']
    valid_bs = configs['training']['valid_bs']
    test_bs = configs['training']['test_bs']
    weights_file_addon = configs['training']['weights_file_addon']
    rename_side = True if 'center_rmBackground' in cam_inputs else False

    if rename_side:
        cam_inputs[-1] = 'Side'
    cam_str = ''.join([c[0].lower() for c in cam_inputs])
    if weights_file_addon:
        weights_file = f"{subnet_name}_{cam_str}cams_{configs['training']['num_fc_hidden_units']}_{weights_file_addon}"
    else:
        weights_file = f"{subnet_name}_{cam_str}cams_{configs['training']['num_fc_hidden_units']}_fully_trained"
    print(f'Loading weights from: {weights_file}')

    if rescale_pos:
        # Set min and max XYZ position values: [[xmin, ymin, zmin], [xmax, ymax, zmax]
        # min_max_pos = [[299, 229, 279], [401, 311, 341]]
        # min_max_pos = [[254, 203, 234], [472, 335, 362]]  # Old Min and max values for all trajectories
        min_max_pos = [[290, 235, 275], [410, 305, 345]]  # Min and max values for all trajectories
        # min_max_pos = utils_data.get_dataset_min_max_pos(configs)
    else:
        min_max_pos = None

    # Set training device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Inference done on {device}')

    # Merge all annotation files based on config file
    print("Loading annotation files...")
    anno_paths = anno_paths_train
    annotations = utils_data.merge_annotations(dataset_root, anno_paths)
    annotations = annotations.head(1)   # Keep only one image

    # Create dataset object
    print("Initializing dataset object...")
    # Create dataset objects
    if rename_side:
        cam_inputs[-1] = 'center_rmBackground'
    if grayscale:
        transforms = v2.Compose([torchvision.transforms.Resize((resize_img_h, resize_img_w)),
                                 torchvision.transforms.Grayscale(),
                                 NormTransform()])  # Remember to also change the annotations for other transforms
    else:
        transforms = v2.Compose([torchvision.transforms.Resize((resize_img_h, resize_img_w)),
                                 NormTransform()])
    dataset = MandibleDataset(dataset_root, cam_inputs, annotations, min_max_pos, transforms)

    print("Creating dataloader...")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Define the model
    print("Loading model...")
    model = SiameseNetwork(configs)
    # Load trained weights
    model.load_state_dict(torch.load(f"siamese_net/model_weights/{weights_file}.pth"))
    model.to(device)

    # Get the subnet sections
    subnet_layers = model.subnet

    with torch.no_grad():
        for img_id, (images, targets) in tqdm(enumerate(dataloader)):
            images = [img.to(device) for img in images]
            # Get the activations at each step of the subnet
            all_outputs = []
            for i, layer in enumerate(subnet_layers):
                if isinstance(layer, torch.nn.Conv2d) | isinstance(layer, torch.nn.Sequential):
                    # print(f'Layer {i}: ')
                    # print(layer)
                    sub_model = torch.nn.Sequential(*(subnet_layers[:i + 1]))
                    sub_model_outputs = [sub_model(img) for img in images]
                    all_outputs.append(sub_model_outputs)
            # Reformat images
            images = [img[0, ...].detach().cpu().numpy() for img in images]
            # outputs = [output[0, ...].detach().cpu().numpy() for sub_outputs in all_outputs for output in sub_outputs]
            images = [(img.transpose((1, 2, 0)) * 255).astype('uint8') for img in images]
            # Overlay feature maps
            heatmaps_imgs = []
            for sub_model_output in all_outputs:
                outputs = [output[0, ...].detach().cpu().numpy() for output in sub_model_output]
                heatmap_cam = overlay_activation_map(images, outputs)
                heatmaps_imgs.append(heatmap_cam)
            # Display results
            fig, axs = plt.subplots(5, 3, figsize=(7, 10), layout='constrained')
            # heatmaps = [np.maximum(np.mean(heatmap, axis=0), 0) / np.max(heatmap) for heatmap in outputs]
            # plt.figure(figsize=(3 * len(heatmap_imgs), 7))
            img_name = dataloader.dataset.img_names[img_id]
            heatmaps_imgs_flat = [h_img for h_img_cam in heatmaps_imgs for h_img in h_img_cam]
            for id, (ax, h_map) in enumerate(zip(axs.flat, heatmaps_imgs_flat)):
                ax.imshow(h_map)
                if id == 0:
                    ax.set_title(f'Left')
                if id == 1:
                    ax.set_title(f'Right')
                if id == 2:
                    ax.set_title(f'Side')
                if id % 3 == 0:
                    ax.set_ylabel(f'{list(all_outputs[int(id/3)][0].shape[-2:])}')
                ax.set_xticks([])
                ax.set_yticks([])
            plt.show()

    print()



