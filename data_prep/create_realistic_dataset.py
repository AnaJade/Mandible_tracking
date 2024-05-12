import argparse
import pathlib

import cv2
import matplotlib
from tqdm import tqdm

from siamese_net import utils_data

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torchvision

# Disable warning for using transforms.v2
torchvision.disable_beta_transforms_warning()
from torchvision.io import read_image

import utils


def create_realistic_image(dataset_root: pathlib.Path, annotations: pd.DataFrame,
                           mandible_colour=(180, 121, 81), pixel_range=10) -> [pd.DataFrame, list]:
    """
    Filter out images where the mandible goes out of frame based on matching mandible pixel values
    Loads the two front images, and checks the border
    :param dataset_root: path to the root of the dataset
    :param annotations: annotations df with all trajectories
    :param mandible_colour: maximum pixel RGB value difference to still be considered as part of the background
    :param pixel_range: Range of values th be
    :return: filtered annotation df
    """
    cameras = {'Left': 'l', 'Right': 'r', 'Side': 's'}
    bgnd_img = read_image(dataset_root.joinpath(f'chair_background.jpg').__str__()).numpy().transpose((1, 2, 0))
    bgnd_img = cv2.resize(bgnd_img, dsize=(1920, 1200), interpolation=cv2.INTER_CUBIC)
    for img_set in tqdm(annotations.index.values.tolist()):
        # Build full image paths
        img_paths = [dataset_root.joinpath(f'{cam}/{img_set}_{cameras[cam]}.jpg') for cam in ['Left', 'Right', 'Side']]
        # Load images
        imgs = [read_image(img_path.__str__()).numpy().transpose((1, 2, 0)) for img_path in img_paths]
        # Crop out mandible
        mandible_crops = utils_data.crop_mandible_by_pixel_match(imgs, mandible_colour, pixel_range)
        # Overlay imgs
        # real_imgs = [cv2.addWeighted(crop, 0.5, bgnd_img, 0.5, 0.0) for crop in mandible_crops]
        real_imgs = []
        for m in mandible_crops:
            m_pos = np.nonzero(m)
            real_img = bgnd_img.copy()
            real_img[max(min(m_pos[0]), 0):min(max(m_pos[0]), 1200),
                     max(min(m_pos[1]), 0):min(max(m_pos[1]), 1920),
                     :] = m[max(min(m_pos[0]), 0):min(max(m_pos[0]), 1200),
                            max(min(m_pos[1]), 0):min(max(m_pos[1]), 1920), :]
            real_imgs.append(real_img)
        fig, axs = plt.subplots(1, 3, layout='tight')
        [axs[i].imshow(real_imgs[i]) for i in range(3)]
        plt.show()


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path',
                        help='Path to the config file',
                        type=str)

    args = parser.parse_args()
    config_file = pathlib.Path(args.config_path)
    # config_file = pathlib.Path("data_prep/data_config.yaml")

    if not config_file.exists():
        print(f'Config file not found at {args.config_path}')
        raise SystemExit(1)
    configs = utils.load_configs(config_file)

    data_folder_path = pathlib.Path(configs['annotations']['data_folder_path'])
    traj_name = configs['annotations']['traj_name']
    traj_folder_path = data_folder_path.joinpath(traj_name)

    dataset_root = pathlib.Path(configs['images']['img_root'])
    anno_files = configs['merge_trajectories']['traj_to_merge']
    new_file_name_base = configs['merge_trajectories']['merged_file_name']
    reduce_rot = configs['merge_trajectories']['filter_rot']
    filter_oof = configs['merge_trajectories']['filter_oof']
    test_ratio = configs['merge_trajectories']['test_ratio']
    valid_ratio = configs['merge_trajectories']['valid_ratio']

    # Merge all annotation files together
    annotations = utils_data.merge_annotations(dataset_root, anno_files)

    pixel_range_windows = [199, 134, 98]
    pixel_range_ubuntu = [180, 121, 81]
    annotations, removed_imgs = create_realistic_image(dataset_root, annotations, pixel_range_windows, 30)
    print(f'Removed {len(removed_imgs)} images')
