import pathlib
import warnings

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

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
        self.cameras = {'Left': 'l', 'Right': 'r', 'Side': 's', 'center_rmBackground': 's'}

        self.pos_min = None
        self.pos_max = None
        if self.min_max_pos is not None:
            self.pos_min = np.array(self.min_max_pos[0])
            self.pos_max = np.array(self.min_max_pos[1])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_id = self.img_names[idx]
        img_pose = self.img_labels.iloc[idx].to_numpy()
        # Get img paths for corresponding frame
        img_paths = [self.root.joinpath(cam).joinpath(f"{img_id}_{self.cameras[cam]}.jpg") for cam in
                     self.cam_inputs]
        # Read images
        try:
            images = [read_image(img_path.__str__()) for img_path in img_paths]
        # Version with raising another exception (stops when an empty file is found)
        except RuntimeError:
            raise FileNotFoundError(f"\nOne of {[img_path.stem for img_path in img_paths]} couldn't be loaded")

        # Re-scale position to be between -1 and 1
        if self.min_max_pos is not None:
            img_pose = normalize_position(torch.Tensor(img_pose), self.pos_min, self.pos_max)

        # Apply transforms
        if self.transforms:
            images = [self.transforms(image) for image in images]

        # Reshape target if needed
        if len(img_pose.shape) > 2:
            img_pose = np.squeeze(img_pose)

        # Version with the warning (runs VERY SLOWLY)
        """
        except RuntimeError:
            # Return empty tensors if the image is unreadable
            # TODO: Find better way to get true image dim
            images = len(img_paths)*[torch.zeros([3, 1200, 1920])]
            img_pose = torch.zeros(1, 7)
            warning_str = f"\nOne of {[img_path.stem for img_path in img_paths]} doesn't exist"
            warnings.warn(warning_str)

        if sum(images[0].flatten()) > 0:
            # Re-scale position to be between -1 and 1
            if self.min_max_pos is not None:
                img_pose = normalize_position(torch.Tensor(img_pose), self.pos_min, self.pos_max)

            # Apply transforms
            if self.transforms:
                images = [self.transforms(image) for image in images]

            # Reshape target if needed
            if len(img_pose.shape) > 2:
                img_pose = np.squeeze(img_pose)
        """
        return images, np.float64(img_pose)


class NormTransform(torch.nn.Module):
    """
    Convert tensor type from uint8 to float32, and divide by 255
    """
    def forward(self, img):
        return img.float()/255


def create_mini_dataset(csv_path: pathlib.Path, mini_ds_size=5):
    """
    Create a mini version of the full annotation file
    :param csv_path: path to the full annotation file
    :param mini_ds_size: number of samples to keep
    :return:
    """
    data = pd.read_csv(csv_path)
    data.head(mini_ds_size).to_csv(csv_path.parent.joinpath(f'{csv_path.stem}_mini.csv'), header=True, index=False)


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


def get_dataset_min_max_pos(config: dict) -> list:
    # Load the annotations for all sets (train, valid, test) and get the min-max values for X, Y, Z
    paths = (config['data']['trajectories_train'] + config['data']['trajectories_valid'] +
             config['data']['trajectories_test'])

    # Merge annotations
    dataset_root = pathlib.Path(config['data']['dataset_root'])
    merged_annos = merge_annotations(dataset_root, paths)
    merged_pos = merged_annos.loc[:, ['x', 'y', 'z']]
    min_pos = list(np.floor(merged_pos.min(axis=0)))
    max_pos = list(np.ceil(merged_pos.max(axis=0)))

    return [min_pos, max_pos]


def filter_imgs_per_position(annotations: pd.DataFrame, axis_lim: list | None, axis=None) -> pd.DataFrame:
    """
    Filter the given annotation dataframe to keep only images that have a pose lying within the given limits
    :param annotations: annotation dataframe
    :param axis_lim: [[xmin, xmax], [ymin, ymax], [zmin, zmax]], leave empty for no limits
    :param axis: list with the axis to restrict. axis_lim will overwrite this value if it is not None
    :return: filtered annotations
    """
    if axis_lim is None and axis is not None:
        # axis_lim_center = [[345, 355], [265, 275], [305, 315]]  # center position +/- 5 mm
        axis_lim_center = [[335, 355], [270, 290], [300, 320]]  # center position +/- 1 cm
        axis_lim_all = [[290, 410], [235, 305], [275, 345]]     # Intended full range
        axis_lim = [axis_lim_center[i] if ax in axis else axis_lim_all[i] for i, ax in enumerate(['x', 'y', 'z'])]
    elif axis_lim is None and axis is None:
        print('No filter parameters were given. Returning the original dataframe...')
        axis_lim = [[], [], []]

    # Filter on x limits
    if axis_lim[0]:
        annotations = annotations[annotations['x'].between(*axis_lim[0])]
    # Filter on y limits
    if axis_lim[1]:
        annotations = annotations[annotations['y'].between(*axis_lim[1])]
    # Filter on z limits
    if axis_lim[2]:
        annotations = annotations[annotations['z'].between(*axis_lim[2])]
    return annotations


def filter_imgs_per_rotation(annotations: pd.DataFrame, rot_lim: list | None) -> pd.DataFrame:
    """
    Filter the given annotation dataframe to keep only images that have a pose lying within the given limits
    :param annotations: annotation dataframe
    :param rot_lim: [[q1min, q1max], [q2min, q2max], [q3min, q3max], [q4min, q4max]], leave empty for no limits
    :return: filtered annotations
    """
    if rot_lim is None:
        # Use default as no rotation
        rot_lim = [[0.499, 0.501], [-0.501, -0.499], [0.499, 0.501], [0.499, 0.501]]
        # rot_lim = [[0, 1], [-1, 0], [0, 1], [0, 1]]

    for lim, q in zip(rot_lim, ['q1', 'q2', 'q3', 'q4']):
        annotations = annotations[annotations[q].between(*lim)]
    return annotations


def filter_imgs_per_rotation_euler(annotations: pd.DataFrame, rot_lim: list | None) -> pd.DataFrame:
    """
    Filter the given annotation dataframe to keep only images that have a pose lying within the given limits
    :param annotations: annotation dataframe (still with quaternions)
    :param rot_lim: [[rxmin, rxmax], [rymin, rymax], [rzmin, rzmax]], leave empty for no limits
    :return: filtered annotations
    """
    if rot_lim is None:
        # Use default as no rotation
        rot_lim = [[-90-5, -90+5], [90-10, 90+10], [-5, 5]]
        # rot_lim = [[0, 1], [-1, 0], [0, 1], [0, 1]]
    # Convert quaternion to eule
    q = annotations[[f'q{i+1}' for i in range(4)]]
    annotations = get_euler_annotations([annotations])[0]
    # Filter
    for lim, r in zip(rot_lim, ['Rx', 'Ry', 'Rz']):
        annotations = annotations[annotations[r].between(*lim)]
    # Replace euler angles with quaternions
    annotations = pd.merge(annotations, q, left_index=True, right_index=True)
    annotations = annotations[~annotations.index.duplicated(keep='first')]
    annotations = annotations.drop(['Rx', 'Ry', 'Rz'], axis=1)
    print(f'Removed {len(q) - len(annotations)} images')
    return annotations


def get_loss_per_axis(targets: np.ndarray, preds: np.ndarray) -> pd.DataFrame:
    """
    Get the RMSE for each dimension
    :param targets: [_, 6 or 7] array with the target values
    :param preds: [_, 6 or 7] array with the model predictions
    :return: pandas df with the RMSE per dimension
    """
    # Return a pandas df with the RMSE in X, Y, Z, rx, r, rz
    assert preds.shape[-1] == targets.shape[-1]
    if preds.shape[-1] == 6:
        dims = ['x', 'y', 'z', 'rx', 'ry', 'rz']
    elif preds.shape[-1] == 7:
        dims = ['x', 'y', 'z', 'q1', 'q2', 'q3', 'q4']
    else:
        dims = []

    rmse_per_dim = {}
    for i, dim in enumerate(dims):
        rmse_per_dim[dim] = mean_squared_error(targets[:, i], preds[:, i], squared=False)

    rmse_per_dim = pd.DataFrame.from_dict(rmse_per_dim, orient='index', columns=['RMSE'])
    return rmse_per_dim


def get_loss_per_img(targets: np.ndarray, preds: np.ndarray) -> pd.DataFrame:
    """
    Get the position, orientation and total RMSE for each image
    :param targets: [_, 6 or 7] array with the target values
    :param preds: [_, 6 or 7] array with the model predictions
    :return: pandas df with the RMSE per image
    """
    # Return a pandas df with the RMSE in X, Y, Z, rx, r, rz
    assert preds.shape[-1] == targets.shape[-1]
    rmse_per_img = {}
    for i in range(preds.shape[0]):
        rmse_per_img[i] = [mean_squared_error(targets[i, :3], preds[i, :3], squared=False),    # RMSE position
                           mean_squared_error(targets[i, 3:], preds[i, 3:], squared=False),    # RMSE orientation
                           mean_squared_error(targets[i, :], preds[i, :], squared=False)]      # RMSE total

    rmse_per_img = pd.DataFrame.from_dict(rmse_per_img, orient='index', columns=['RMSE_pos', 'RMSE_ori', 'RMSE'])
    return rmse_per_img


def get_pcc_per_axis(annotations: pd.DataFrame, preds: pd.DataFrame, angle='Euler') -> pd.DataFrame:
    """
    Get the Pearson product-moment correlation coefficients per axis
    :param annotations: annotations dataframe
    :param preds: prediction dataframe
    :param angle: whether to use quaternions or Euler angles
    :return: dataframe with the PCC per axis
    """
    if angle == 'Euler':
        annotations = get_euler_annotations([annotations])[0]
        preds = get_euler_annotations([preds])[0]

    pcc_per_axis = {}
    for ax in annotations.columns:
        pcc_per_axis[ax] = np.corrcoef(annotations[ax].to_numpy(), preds[ax].to_numpy(), rowvar=False)[0, 1]
    pcc_per_axis = pd.DataFrame.from_dict(pcc_per_axis, orient='index', columns=['PCC'])
    return pcc_per_axis


if __name__ == '__main__':
    config_file = pathlib.Path("siamese_net/config_windows.yaml")
    configs = utils.load_configs(config_file)
    dataset_root = pathlib.Path(configs['data']['dataset_root'])
    anno_paths_train = configs['data']['trajectories_train']
    anno_paths_valid = configs['data']['trajectories_valid']
    anno_paths_test = configs['data']['trajectories_test']

    cam_inputs = configs['training']['cam_inputs']
    train_bs = configs['training']['train_bs']
    valid_bs = configs['training']['valid_bs']
    test_bs = configs['training']['test_bs']

    # Get min-max position for all annotation files
    min_max_pos = get_dataset_min_max_pos(configs)

    # Merge all annotation files based on config file
    print("Loading annotation files...")
    annotations_train = merge_annotations(dataset_root, anno_paths_train)
    annotations_valid = merge_annotations(dataset_root, anno_paths_valid)
    annotations_test = merge_annotations(dataset_root, anno_paths_test)

    """
    Test rot filter function
    """
    train_reduce_rot = filter_imgs_per_rotation_euler(annotations_train, None)
    print(len(annotations_train))
    print(len(train_reduce_rot))

    # Create dataset object
    print("Initializing dataset object...")
    # Create dataset objects
    transforms = v2.Compose([NormTransform()])  # Remember to also change the annotations for other transforms
    dataset_train = MandibleDataset(dataset_root, cam_inputs, annotations_train, None, transforms)
    dataset_valid = MandibleDataset(dataset_root, cam_inputs, annotations_valid, None, transforms)
    dataset_test = MandibleDataset(dataset_root, cam_inputs, annotations_test, None, transforms)

    print("Creating dataloader...")
    dataloader_train = DataLoader(dataset_train, batch_size=train_bs, shuffle=False, num_workers=0)
    dataloader_valid = DataLoader(dataset_valid, batch_size=valid_bs, shuffle=False, num_workers=0)
    dataloader_test = DataLoader(dataset_test, batch_size=test_bs, shuffle=False, num_workers=0)

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
