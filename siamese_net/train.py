import argparse
import pathlib
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
    grayscale = configs['data']['grayscale']
    rescale_pos = configs['data']['rescale_pos']

    subnet_name = configs['training']['sub_model']
    weights_file_addon = configs['training']['weights_file_addon']
    cam_inputs = configs['training']['cam_inputs']
    train_bs = configs['training']['train_bs']
    valid_bs = configs['training']['valid_bs']
    test_bs = configs['training']['test_bs']

    nb_epochs = configs['training']['max_epochs']
    patience = configs['training']['patience']
    lr = configs['training']['lr']
    scheduler_step_size = configs['training']['lr_scheduler']['step_size']
    scheduler_gamma = configs['training']['lr_scheduler']['gamma']

    wandb_log = configs['wandb']['wandb_log']
    project_name = configs['wandb']['project_name']

    cam_str = ''.join([c[0].lower() for c in cam_inputs])
    if weights_file_addon:
        weights_file = f"{subnet_name}_{cam_str}cams_{configs['training']['num_fc_hidden_units']}_{weights_file_addon}"
    else:
        weights_file = f"{subnet_name}_{cam_str}cams_{configs['training']['num_fc_hidden_units']}"
    print(f'Weights will be saved to: {weights_file}')

    if rescale_pos:
        # Set min and max XYZ position values: [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        # min_max_pos = [[299, 229, 279], [401, 311, 341]]
        min_max_pos = utils_data.get_dataset_min_max_pos(configs)
    else:
        min_max_pos = None

    # Set training device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Training done on {device}')

    # Merge all annotation files based on config file
    print("Loading annotation files...")
    annotations_train = utils_data.merge_annotations(dataset_root, anno_paths_train)
    annotations_valid = utils_data.merge_annotations(dataset_root, anno_paths_valid)
    annotations_test = utils_data.merge_annotations(dataset_root, anno_paths_test)

    # Create dataset object
    print("Initializing dataset object...")
    # Create dataset objects
    if grayscale:
        transforms = v2.Compose([torchvision.transforms.Resize((resize_img_h, resize_img_w)),
                                 torchvision.transforms.Grayscale(),
                                 NormTransform()])  # Remember to also change the annotations for other transforms
    else:
        transforms = v2.Compose([torchvision.transforms.Resize((resize_img_h, resize_img_w)),
                                 NormTransform()])
    dataset_train = MandibleDataset(dataset_root, cam_inputs, annotations_train, min_max_pos, transforms)
    dataset_valid = MandibleDataset(dataset_root, cam_inputs, annotations_valid, min_max_pos, transforms)
    dataset_test = MandibleDataset(dataset_root, cam_inputs, annotations_test, min_max_pos, transforms)

    print("Creating dataloader...")
    # dataloader_train = DataLoader(dataset_train, batch_size=train_bs, shuffle=True, num_workers=4)
    # dataloader_valid = DataLoader(dataset_valid, batch_size=valid_bs, shuffle=False, num_workers=4)
    # dataloader_test = DataLoader(dataset_test, batch_size=test_bs, shuffle=False, num_workers=4)
    dataloader_train = DataLoader(dataset_train, batch_size=train_bs, shuffle=False, num_workers=0)
    dataloader_valid = DataLoader(dataset_valid, batch_size=valid_bs, shuffle=False, num_workers=0)
    dataloader_test = DataLoader(dataset_valid, batch_size=test_bs, shuffle=False, num_workers=0)

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
    model_fit = train_model(configs, model, [dataloader_train, dataloader_valid, dataloader_test],
                            device, criterion, optimizer, scheduler)

    # Save model (not needed, since the model is saved during training
    """
    print("Saving best model...")
    # weights_file = f"{subnet_name}_{cam_str}cams_{configs['training']['num_fc_hidden_units']}_{weights_file_addon}"
    if weights_file_addon:
        weights_file = f"{subnet_name}_{cam_str}cams_{configs['training']['num_fc_hidden_units']}_{weights_file_addon}"
    else:
        weights_file = f"{subnet_name}_{cam_str}cams_{configs['training']['num_fc_hidden_units']}"
    torch.save(model_fit.state_dict(), f"siamese_net/model_weights/{weights_file}.pth")
    """


