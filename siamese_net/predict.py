import argparse
import pathlib

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
from SiameseNet import SiameseNetwork, get_preds


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
    anno_paths_test = configs['data']['trajectories_test']
    rescale_pos = configs['data']['rescale_pos']

    subnet_name = configs['training']['sub_model']
    cam_inputs = configs['training']['cam_inputs']
    test_bs = configs['training']['test_bs']

    if rescale_pos:
        # Set min and max XYZ position values: [[xmin, ymin, zmin], [xmax, ymax, zmax]
        min_max_pos = [[299, 229, 279], [401, 311, 341]]
    else:
        min_max_pos = None

    # Set training device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Inference done on {device}')

    # Merge all annotation files based on config file
    print("Loading annotation files...")
    annotations_test = utils_data.merge_annotations(dataset_root, anno_paths_test)

    # Create dataset object
    print("Creating dataloader...")
    # Create dataset objects
    transforms = v2.Compose([NormTransform()])  # Remember to also change the annotations for other transforms
    dataset_test = MandibleDataset(dataset_root, cam_inputs, annotations_test, min_max_pos, transforms)
    # NOTE: shuffle has to be false, to be able to match the predictions to the right frames
    dataloader_test = DataLoader(dataset_test, batch_size=test_bs, shuffle=False, num_workers=4)

    # Define the model
    print("Loading model...")
    model = SiameseNetwork(configs)
    # Load trained weights
    weights_file = f"{subnet_name}_{len(cam_inputs)}cams_{configs['training']['num_fc_hidden_units']}"
    model.load_state_dict(torch.load(f"siamese_net/model_weights/{weights_file}.pth"))
    model.to(device)

    print("Performing inference...")
    preds = get_preds(model, device, dataloader_test, min_max_pos)

    # Calculate the loss
    test_loss = mean_squared_error(annotations_test.to_numpy(), preds.to_numpy())
    print(f'Test loss: {test_loss}')

    # Calculate the loss on the normalized data
    norm_annotations = utils_data.normalize_position(torch.Tensor(annotations_test.to_numpy()),
                                                     np.array(min_max_pos[0]), np.array(min_max_pos[1]))
    norm_preds = utils_data.normalize_position(torch.Tensor(preds.to_numpy()),
                                               np.array(min_max_pos[0]), np.array(min_max_pos[1]))
    test_loss = mean_squared_error(norm_annotations, norm_preds)
    print(f'Test loss on normalized data: {test_loss}')

    # Format to pandas df
    preds_df = annotations_test.copy()
    preds_df.iloc[:, :] = preds

    # Save preds as csv
    print("Saving results...")
    preds_file = f"{subnet_name}_{len(cam_inputs)}cams_{configs['training']['num_fc_hidden_units']}"
    preds_df.to_csv(f"siamese_net/preds/{preds_file}.csv")

