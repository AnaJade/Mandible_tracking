import argparse
import pathlib
import platform
import random

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


def create_realistic_image(dataset_root: pathlib.Path, annotations: pd.DataFrame, bgnd_imgs: list[np.ndarray] | None,
                           mandible_colour=(180, 121, 81), pixel_range=10) -> None:
    """
    Filter out images where the mandible goes out of frame based on matching mandible pixel values
    Loads the two front images, and checks the border
    :param dataset_root: path to the root of the dataset
    :param annotations: annotations df with all trajectories
    :param bgnd_imgs: list of possible background images as RGB [1200 x 1920] np array
    :param mandible_colour: maximum pixel RGB value difference to still be considered as part of the background
    :param pixel_range: Range of pixel values to be kept
    :return:
    """
    cameras = {'Left': 'l', 'Right': 'r', 'Side': 's'}
    cropped_pos = {}
    for img_set in tqdm(annotations.index.values.tolist()):
        # Build full image paths
        img_paths = [dataset_root.joinpath(f'{cam}/{img_set}_{cameras[cam]}.jpg') for cam in ['Left', 'Right', 'Side']]
        # Load images
        imgs = [read_image(img_path.__str__()).numpy().transpose((1, 2, 0)) for img_path in img_paths]
        # Crop out mandible
        mandible_crops, crop_pos = utils_data.crop_mandible_by_pixel_match(imgs, mandible_colour, pixel_range)
        # Save crop coords
        for i, cam in enumerate(['Left', 'Right', 'Side']):
            cropped_pos[f'{img_set}_{cameras[cam]}'] = crop_pos[i]
        # Overlay imgs
        if bgnd_imgs is not None:
            real_imgs = []
            bgnd_img = random.choice(bgnd_imgs)
            for m, c_pos in zip(mandible_crops, crop_pos):
                real_img = bgnd_img.copy()
                real_img[c_pos['h_min']:c_pos['h_max'],
                        c_pos['w_min']:c_pos['w_max'], :] = m[c_pos['h_min']:c_pos['h_max'],
                                                            c_pos['w_min']:c_pos['w_max'], :]
                real_imgs.append(real_img)
        else:
            real_imgs = mandible_crops
        """
        fig, axs = plt.subplots(1, 3, layout='tight')
        [axs[i].imshow(real_imgs[i]) for i in range(3)]
        plt.show()
        """
        # Save new imgs
        img_paths_new = [dataset_root.joinpath(f'{cam}_crop/{img_set}_{cameras[cam]}.jpg') for
                         cam in ['Left', 'Right', 'Side']]
        [cv2.imwrite(img_file.__str__(), cv2.cvtColor(img, cv2.COLOR_RGB2BGR)) for
         img_file, img in zip(img_paths_new, real_imgs)]
    # Save cropped pos
    cropped_pos = pd.DataFrame.from_dict(cropped_pos, orient='index').reset_index().rename(columns={'index': 'img'})
    cropped_pos = cropped_pos.set_index('img')
    cropped_pos_file = dataset_root.joinpath(f'mandible_crop_coords.csv')
    if cropped_pos_file.exists():
        old_cropped_pos = pd.read_csv(cropped_pos_file, index_col='img')
        cropped_pos = pd.concat([old_cropped_pos, cropped_pos], axis=0).drop_duplicates(keep='last')
    cropped_pos.to_csv(cropped_pos_file)


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

    dataset_root = pathlib.Path(configs['images']['img_root'])
    bgnd_img_paths = configs['add_real_bgnd']['bgnd_img_paths']
    trajectories = configs['add_real_bgnd']['trajectory_annotations']
    bgnd_imgs_paths = configs['add_real_bgnd']['bgnd_img_paths']

    # Merge all annotation files together
    annotations = utils_data.merge_annotations(dataset_root, trajectories)

    # Load all possible background images
    bgnd_imgs = [read_image(dataset_root.joinpath(bgnd_img_path).__str__()).numpy().transpose((1, 2, 0)) for
                 bgnd_img_path in bgnd_imgs_paths]
    bgnd_imgs = [cv2.resize(bgnd_img, dsize=(1920, 1200), interpolation=cv2.INTER_CUBIC) for bgnd_img in bgnd_imgs]

    if platform.system() == 'Windows':
        pixel_range = [199, 134, 98]
    else:
        pixel_range = [180, 121, 81]

    create_realistic_image(dataset_root, annotations, None, pixel_range, 30)
    print()
