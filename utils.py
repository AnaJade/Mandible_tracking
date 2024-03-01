import pathlib
import yaml
import numpy as np
import pandas as pd

from scipy.spatial.transform import Rotation


def load_configs(config_path: pathlib.Path) -> dict:
    """
    Load the configs in the yaml file, and returns them in a dictionary
    :param config_path: path to the config file as a pathlib path
    :return: dict with all configs in the file
    """
    # Read yaml config
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)

    return configs


def load_frame_time_steps(csv_path: pathlib.Path) -> pd.DataFrame:
    """
    Load and format the image time frame csv file
    :param csv_path: path to the time_steps csv file
    :return: dataframe with the file data
    """
    csv = pd.read_csv(csv_path, header=None, names=['time'])
    return csv


def load_log_file(txt_path: pathlib.Path) -> pd.DataFrame:
    """
    Load and format the robot position log txt file
    :param txt_path: path to the robot pose log file
    :return: dataframe with the file data
    """
    col_names = ['time', 'x', 'y', 'z', 'q1', 'q2', 'q3', 'q4']
    txt = pd.read_table(txt_path, sep=',', header=None, names=col_names)
    return txt


def quaternion2euler(q: np.array) -> np.array:
    """
    Convert a 1D or 2D quaternion np array to the equivalent  [Z, Y, X] euler angles
    :param q: quaternions, as either a 1D or 2D array
    :return: euler angle equivalent for the given quaternions
    """
    reshape_output = False
    if np.isnan(q).any():
        rot_euler = np.empty(3)
        rot_euler[:] = np.nan
    else:
        # Scipy uses real last, so change the order
        if len(q.shape) == 1:
            reshape_output = True
            q = np.expand_dims(q, 0)
        q = q[:, [1, 2, 3, 0]]
        rot = Rotation.from_quat(q)
        rot_euler = rot.as_euler('zyx', degrees=True)
        # Reformat to get [Rx, Ry, Rz] based on the robot
        rot_euler = rot_euler[:, [2, 0, 1]]
        if reshape_output:
            rot_euler = np.squeeze(rot_euler, axis=0)
    return rot_euler


if __name__ == '__main__':
    config_file = pathlib.Path("data_prep/data_config.yaml")
    configs = load_configs(config_file)
    data_folder_path = pathlib.Path(configs['data_folder_path'])
    traj_name = configs['traj_name']
    traj_folder_path = data_folder_path.joinpath(traj_name)
    log_file = traj_folder_path.joinpath(configs['log_name'])
    frame_file = traj_folder_path.joinpath("time_stamps.csv")

    print(configs)
    print(log_file)
    print(frame_file)

    # Load files
    traj = load_log_file(log_file)
    frame_timings = load_frame_time_steps(frame_file)

    print()
