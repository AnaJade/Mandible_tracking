import argparse
import pathlib

import numpy as np
import pandas as pd
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
    weights_file_addon = configs['training']['weights_file_addon']

    cam_str = ''.join([c[0].lower() for c in cam_inputs])
    if weights_file_addon:
        weights_file = f"{subnet_name}_{cam_str}cams_{configs['training']['num_fc_hidden_units']}_{weights_file_addon}"
    else:
        weights_file = f"{subnet_name}_{cam_str}cams_{configs['training']['num_fc_hidden_units']}"
    print(f'Loading weights from: {weights_file}')

    if rescale_pos:
        # Set min and max XYZ position values: [[xmin, ymin, zmin], [xmax, ymax, zmax]
        # min_max_pos = [[299, 229, 279], [401, 311, 341]]
        min_max_pos = [[254, 203, 234], [472, 335, 362]]    # Min and max values for all trajectories
        # min_max_pos = utils_data.get_dataset_min_max_pos(configs)
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
    # dataloader_test = DataLoader(dataset_test, batch_size=test_bs, shuffle=False, num_workers=4)
    dataloader_test = DataLoader(dataset_test, batch_size=test_bs, shuffle=False, num_workers=0)

    # Define the model
    print("Loading model...")
    model = SiameseNetwork(configs)
    # Load trained weights
    model.load_state_dict(torch.load(f"siamese_net/model_weights/{weights_file}.pth"))
    model.to(device)

    print("Performing inference...")
    preds = get_preds(model, device, dataloader_test, min_max_pos)

    # Calculate the loss
    test_rmse = mean_squared_error(annotations_test.to_numpy(), preds.to_numpy(), squared=False)
    print(f'Test RMSE: {test_rmse}')

    # Calculate the loss per dimension
    # annotations_test_euler = utils.pose_quaternion2euler(annotations_test.to_numpy())
    # preds_euler = utils.pose_quaternion2euler(preds.to_numpy())
    rmse_per_dim = utils_data.get_loss_per_axis(annotations_test.to_numpy(), preds.to_numpy())
    print(f'RMSE per dimension: \n{rmse_per_dim}')

    # Calculate the loss on the normalized data
    norm_annotations = utils_data.normalize_position(torch.Tensor(annotations_test.to_numpy()),
                                                     np.array(min_max_pos[0]), np.array(min_max_pos[1]))
    norm_preds = utils_data.normalize_position(torch.Tensor(preds.to_numpy()),
                                               np.array(min_max_pos[0]), np.array(min_max_pos[1]))
    test_rmse = mean_squared_error(norm_annotations, norm_preds, squared=False)
    print(f'Test RMSE on normalized data: {test_rmse}')

    # Calculate the rotation error

    rot_q_diff = utils.hamilton_prod(annotations_test.to_numpy()[:, -4:], preds.to_numpy()[:, -4:])

    # Convert error quaternion to Euler
    rot_euler_error = utils.quaternion2euler(rot_q_diff)
    rot_euler_avg_err = pd.DataFrame(np.mean(rot_euler_error, axis=0), index=['Rx_err', 'Ry_err', 'Rz_err'],
                                     columns=['Rot_err'])
    print(f'Average orientation error:\n{rot_euler_avg_err}')

    # Format to pandas df
    preds_df = annotations_test.copy()
    preds_df.iloc[:, :] = preds

    # Append position difference
    pos_diff = annotations_test.to_numpy()[:, :3] - preds.to_numpy()[:, :3]
    pos_diff = pd.DataFrame(pos_diff, columns=['delta_x', 'delta_y', 'delta_z'], index=preds_df.index)
    preds_df = pd.concat([preds_df, pos_diff], axis=1)

    # Append the position, orientation and total RMSE for each image
    rmse_per_image = utils_data.get_loss_per_img(annotations_test.to_numpy(), preds.to_numpy())
    rmse_per_image.index = preds_df.index
    preds_df = pd.concat([preds_df, rmse_per_image], axis=1)

    # Add euler error for each image
    rot_err_per_image = pd.DataFrame(rot_euler_error, columns=['Rx_err', 'Ry_err', 'Rz_err'])
    rot_err_per_image.index = preds_df.index
    preds_df = pd.concat([preds_df, rot_err_per_image], axis=1)

    # Save preds as csv
    print("Saving results...")
    preds_df.to_csv(f"siamese_net/preds/{weights_file}.csv")
    print(preds_df.head(5))

