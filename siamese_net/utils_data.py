import pathlib

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
# Disable warning for using transforms.v2
torchvision.disable_beta_transforms_warning()
from torchvision.io import read_image
from torchvision.transforms import v2

import utils


class MandibleDataset(Dataset):
    def __init__(self, root: pathlib.Path, cam_inputs: list, img_labels: pd.DataFrame, min_max_pos=None,
                 transforms=None):
        """
        Dataset object used to pass images to a siamese network
        :param root: dataset root path
        :param cam_inputs: choose the camera images to use
        :param img_labels: annotations for the images
        :param min_max_pos: min and max position values. If None, no rescaling will be done
        :param transforms: transforms that will be applied to each image before being used as input by the model
        """
        self.root = root
        self.img_labels = img_labels
        self.min_max_pos = min_max_pos
        self.transforms = transforms
        self.img_names = img_labels.index.values.tolist()
        self.cam_inputs = cam_inputs
        self.cameras = {'Left': 'l', 'Right': 'r', 'Side': 's'}

        self.pos_min = None
        self.pos_max = None
        if self.min_max_pos is not None:
            self.pos_min = np.array(self.min_max_pos[0])
            self.pos_max = np.array(self.min_max_pos[1])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):git 
        img_id = self.img_names[idx]
        img_pose = self.img_labels.iloc[idx].to_numpy().astype(float)
        # Get img paths for corresponding frame
        img_paths = [self.root.joinpath(cam).joinpath(f"{img_id}_{self.cameras[cam]}.jpg") for cam in
                     self.cam_inputs]
        # Read images
        images = [read_image(img_path.__str__()) for img_path in img_paths]

        # Re-scale position to be between -1 and 1
        if self.min_max_pos is not None:
            img_pose = normalize_position(torch.Tensor(img_pose), self.pos_min, self.pos_max)

        # Apply transforms
        if self.transforms:
            images = [self.transforms(image) for image in images]
        return images, img_pose


class NormTransform(torch.nn.Module):
    """
    Convert tensor type from uint8 to float32, and divide by 255
    """
    def forward(self, img):
        return img.float()/255


def merge_annotations(dataset_root: pathlib.Path, anno_files: list) -> pd.DataFrame:
    """
    Merge the annotations from each of the desired trajectories into one dataframe
    :param dataset_root: path to the root of the dataset
    :param anno_files: list of annotation files to be merged
    :return:
    """
    # Create full paths
    full_paths = [dataset_root.joinpath(anno_file) for anno_file in anno_files]

    # Read annotation files
    anno_per_file = [pd.read_csv(anno_path, index_col='frame') for anno_path in full_paths]

    # Merge files
    annotations = pd.concat(anno_per_file, axis=0)

    return annotations


def get_euler_annotations(quaternion_annos: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
    Convert the pose quaternions to euler angles
    :param quaternion_annos: list of pose annotation dataframes with quaternions
    :return: list of pose annotation dataframes with euler angles
    """
    e_anno = []
    for q_anno in quaternion_annos:
        # Convert quaternion to euler angle
        q = q_anno.to_numpy()[:, -4:]
        r_euler = utils.quaternion2euler(q)

        # Format dataframe
        pose_euler = q_anno.drop([f'q{i}' for i in range(1, 5)], axis=1)
        pose_euler.loc[:, 'Rx'] = r_euler[:, 0]
        pose_euler.loc[:, 'Ry'] = r_euler[:, 1]
        pose_euler.loc[:, 'Rz'] = r_euler[:, 2]
        e_anno.append(pose_euler)

    return e_anno


def normalize_position(pose: torch.Tensor, min_pos: np.ndarray, max_pos: np.ndarray) -> torch.Tensor:
    """
    Normalize the position values to [-1, 1]
    :param pose: pose values
    :param min_pos: min position values in x, y, z as a np array: [xmin, ymin, zmin]
    :param max_pos: max position values in x, y, z as a np array: [xmax, ymax, zmax]
    :return: Pose with the normalized position values
    """
    # Reshape arrays if necessary
    if len(pose.shape) < 2:
        pose = pose[None, :]
    if len(min_pos.shape) < 2:
        min_pos = np.expand_dims(min_pos, 0)
    if len(max_pos.shape) < 2:
        max_pos = np.expand_dims(max_pos, 0)
    position = pose[:, :3]
    position = 2 * ((position - min_pos) / (max_pos - min_pos)) - 1
    pose[:, :3] = position

    return pose


def rescale_position(pose: torch.Tensor, min_pos: np.ndarray, max_pos: np.ndarray) -> torch.Tensor:
    """
    Rescale the position values from between [-1, 1] to their actual values
    :param pose: scaled pose values
    :param min_pos: min position values in x, y, z as a np array: [xmin, ymin, zmin]
    :param max_pos: max position values in x, y, z as a np array: [xmax, ymax, zmax]
    :return: Pose with the rescaled position values
    """
    positions = pose[:, :3]
    positions = ((positions + 1) / 2) * (max_pos - min_pos) + min_pos
    pose[:, :3] = positions
    return pose


if __name__ == '__main__':
    config_file = pathlib.Path("siamese_net/config.yaml")
    configs = utils.load_configs(config_file)
    dataset_root = pathlib.Path(configs['data']['dataset_root'])
    anno_paths_train = configs['data']['trajectories_train']
    anno_paths_valid = configs['data']['trajectories_valid']
    anno_paths_test = configs['data']['trajectories_test']

    cam_inputs = configs['training']['cam_inputs']
    train_bs = configs['training']['train_bs']
    valid_bs = configs['training']['valid_bs']
    test_bs = configs['training']['test_bs']

    # Merge all annotation files based on config file
    print("Loading annotation files...")
    annotations_train = merge_annotations(dataset_root, anno_paths_train)
    annotations_valid = merge_annotations(dataset_root, anno_paths_valid)
    annotations_test = merge_annotations(dataset_root, anno_paths_test)

    # Create dataset object
    print("Initializing dataset object...")
    # Create dataset objects
    transforms = v2.Compose([NormTransform()])  # Remember to also change the annotations for other transforms
    dataset_train = MandibleDataset(dataset_root, cam_inputs, annotations_train, None, transforms)
    dataset_valid = MandibleDataset(dataset_root, cam_inputs, annotations_valid, None, transforms)
    dataset_test = MandibleDataset(dataset_root, cam_inputs, annotations_test, None, transforms)

    print("Creating dataloader...")
    dataloader_train = DataLoader(dataset_train, batch_size=train_bs, shuffle=False, num_workers=4)
    dataloader_valid = DataLoader(dataset_valid, batch_size=valid_bs, shuffle=False, num_workers=4)
    dataloader_test = DataLoader(dataset_test, batch_size=test_bs, shuffle=False, num_workers=4)

    print("View position and orientation distribution in the dataset")
    # Convert quaternions to euler angles
    annotations_euler = get_euler_annotations([annotations_train, annotations_valid, annotations_test])
    # Plot results
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax_pos = fig.add_subplot(1, 2, 1, projection='3d')
    ax_ori = fig.add_subplot(1, 2, 2, projection='3d')
    colours = ['blue', 'green', 'red']
    markers = ['1', '2', '3']
    handles = []
    for i, ds in enumerate(annotations_euler):
        # Position distribution
        plt_pos = ax_pos.scatter(ds.loc[:, 'x'], ds.loc[:, 'y'], ds.loc[:, 'z'], color=colours[i], marker=markers[i])
        ax_pos.set_title('Position distribution')
        # Orientation distribution
        plt_ori = ax_ori.scatter(ds.loc[:, 'Rx'], ds.loc[:, 'Ry'], ds.loc[:, 'Rz'], color=colours[i], marker=markers[i])
        ax_ori.set_title('Orientation distribution')
        handles.append(plt_pos)
    ax_pos.set_xlabel('X')
    ax_pos.set_ylabel('Y')
    ax_pos.set_zlabel('Z')
    ax_ori.set_xlabel('Rx')
    ax_ori.set_ylabel('Ry')
    ax_ori.set_zlabel('Rz')
    ax_ori.legend(handles=handles, labels=['Train', 'Valid', 'Test'],
                  bbox_to_anchor=(0.4, -0.1), ncol=3)
    plt.suptitle('Pose distribution over all datasets')
    # plt.tight_layout()
    plt.show(block=True)

    print("Fetching data...")
    for imgs, lbl in dataloader_train:
        # Keep only first image in the batch
        imgs = [img[0, :, :, :].squeeze(0) for img in imgs]
        lbl = lbl[0, :].squeeze(0).numpy()

        # Convert quaternion to Euler
        euler = np.round(utils.quaternion2euler(lbl[-4:]), 3)

        # Plot images
        plt.figure(figsize=(3*len(cam_inputs), 3))
        for i, (img, cam) in enumerate(zip(imgs, cam_inputs)):
            img = (img.numpy().transpose((1, 2, 0)) * 255).astype('uint8')
            plt.subplot(1, len(cam_inputs), i+1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(cam)
        plt_title = f'Position: {np.round(lbl[:3], 3)}\nRx: {euler[0]}   Ry: {euler[1]}    Rz: {euler[2]}'
        plt.suptitle(plt_title)
        plt.tight_layout()
        plt.show(block=True)
