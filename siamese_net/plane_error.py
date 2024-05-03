import argparse
import pathlib
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
from SiameseNet import SiameseNetwork, get_plane_error


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path',
                        help='Path to the config file',
                        type=str)

    args = parser.parse_args()
    config_file = pathlib.Path(args.config_path)
    # config_file = pathlib.Path("siamese_net/config.yaml")

    if not config_file.exists():
        print(f'Config file not found at {args.config_path}')
        raise SystemExit(1)

    # Load configs
    configs = utils.load_configs(config_file)
    dataset_root = pathlib.Path(configs['data']['dataset_root'])
    anno_paths_train = configs['data']['trajectories_train']
    anno_paths_valid = configs['data']['trajectories_valid']
    anno_paths_test = configs['data']['trajectories_test']
    resize_img_h = configs['data']['resize_img']['img_h']
    resize_img_w = configs['data']['resize_img']['img_w']
    rescale_pos = configs['data']['rescale_pos']

    subnet_name = configs['training']['sub_model']
    cam_inputs = configs['training']['cam_inputs']
    train_bs = configs['training']['train_bs']
    valid_bs = configs['training']['valid_bs']
    test_bs = configs['training']['test_bs']
    weights_file_addon = configs['training']['weights_file_addon']

    cam_str = ''.join([c[0].lower() for c in cam_inputs])
    if weights_file_addon:
        weights_file = f"{subnet_name}_{cam_str}cams_{configs['training']['num_fc_hidden_units']}_{weights_file_addon}"
    else:
        weights_file = f"{subnet_name}_{cam_str}cams_{configs['training']['num_fc_hidden_units']}_fully_trained"
    print(f'Loading weights from: {weights_file}')

    if rescale_pos:
        # Set min and max XYZ position values: [[xmin, ymin, zmin], [xmax, ymax, zmax]
        # min_max_pos = [[299, 229, 279], [401, 311, 341]]
        # min_max_pos = [[254, 203, 234], [472, 335, 362]]
        min_max_pos = utils_data.get_dataset_min_max_pos(configs)
    else:
        min_max_pos = None

    # Set training device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Inference done on {device}')

    # Merge all annotation files based on config file
    print("Loading annotation files...")
    annotations_train = utils_data.merge_annotations(dataset_root, anno_paths_train)
    annotations_valid = utils_data.merge_annotations(dataset_root, anno_paths_valid)
    annotations_test = utils_data.merge_annotations(dataset_root, anno_paths_test)

    """
    # Create dataset object
    print("Initializing dataset object...")
    # Create dataset objects
    transforms = v2.Compose([NormTransform()])  # Remember to also change the annotations for other transforms
    dataset_train = MandibleDataset(dataset_root, cam_inputs, annotations_train, None, transforms)
    dataset_valid = MandibleDataset(dataset_root, cam_inputs, annotations_valid, None, transforms)
    dataset_test = MandibleDataset(dataset_root, cam_inputs, annotations_test, None, transforms)

    print("Creating dataloader...")
    dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=0)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False, num_workers=0)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    """

    # Define the model
    print("Loading model...")
    model = SiameseNetwork(configs)
    # Load trained weights
    model.load_state_dict(torch.load(f"siamese_net/model_weights/{weights_file}.pth"))
    model.to(device)

    # Filter options
    use_center_only = True  # whether to use images where the position of the mandible lies in the center plane
    # axis_lim = [[358, 368], [264, 274], [293, 303]]  # Middle position based on the data (+/- 5mm)
    # axis_lim = [[353, 373], [259, 279], [288, 308]]  # Middle position based on the data (+/- 1cm)
    axis_lim_mid = [[340, 360], [260, 280], [300, 320]]  # True middle position (+/- 1cm)
    axis_lim_full = [[290, 410], [235, 305], [275, 345]]  # Intended full range
    no_rot = False   # use only images with no rotations
    rot_lim = [[0.499, 0.501], [-0.501, -0.499], [0.499, 0.501], [0.499, 0.501]]
    # Other options
    annotations = annotations_test
    plane = 'xz'
    grid_size = 8
    print("Filtering the data...")
    # Filter images in the dataloader to keep the ones in the center plane
    if use_center_only:
        annotations = utils_data.filter_imgs_per_position(annotations, None,
                                                          axis=[ax for ax in ['x', 'y', 'z'] if ax not in [*plane]])
    if no_rot:
        annotations = utils_data.filter_imgs_per_rotation(annotations, None)
    
    annotations = utils_data.filter_imgs_per_position(annotations, axis_lim_full, None)

    print("Creating the dataloader...")
    transforms = v2.Compose([torchvision.transforms.Resize((resize_img_h, resize_img_w)),
                             NormTransform()])  # Remember to also change the annotations for other transforms
    dataset = MandibleDataset(dataset_root, cam_inputs, annotations, min_max_pos, transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print("Calculating the plane error...")
    if len(dataloader) > 1:
        get_plane_error(model, device, dataloader, min_max_pos, plane, grid_size, annotations_train)
    else:
        print("No images left in the dataset after filtering")

