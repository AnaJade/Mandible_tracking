import pathlib
from tqdm import tqdm

import numpy as np
import pandas as pd
import cv2

import utils


def overlay_pose_quaternions(frame: np.array, pose: pd.DataFrame):
    for i in range(len(pose)):
        text = f"{pose.index[i]}: {np.round(pose[i], 5)}"
        cv2.putText(frame, text, (2900, 1600 + i*100), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (255, 255, 255), 3)
    """
    cv2.imshow("frame", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    return frame


def overlay_pose_euler(frame: np.array, pose: pd.DataFrame):
    # Convert quat to euler
    rot = utils.quaternion2euler(pose[-4:])     # [Ry, Rz, Rx]
    # Reformat pose df
    pose.drop([f'q{i}' for i in range(1, 5)])
    pose = pose[:3]
    pose.loc['Rx'] = rot[0]     # 2, without the last reshaping in quaternion2euler
    pose.loc['Ry'] = rot[1]     # 0
    pose.loc['Rz'] = rot[2]     # 1
    for i in range(len(pose)):
        text = f"{pose.index[i]}: {np.round(pose[i], 5)}"
        cv2.putText(frame, text, (2900, 1600 + i*100), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (255, 255, 255), 3)
    return frame


if __name__ == '__main__':
    # Options
    rot_formats = ['quaternion', 'euler']
    rot_format = rot_formats[1]

    # Extract info from config file and build relevant paths
    config_file = pathlib.Path("data_prep/data_config.yaml")
    configs = utils.load_configs(config_file)

    data_folder_path = pathlib.Path(configs['annotations']['data_folder_path'])
    traj_name = configs['annotations']['traj_name']
    traj_folder_path = data_folder_path.joinpath(traj_name)
    frame_pose_path = traj_folder_path.joinpath("frame_pose_shortcut.csv")  # Remove shortcut when using slerp
    frame_pose = pd.read_csv(frame_pose_path)
    vid_og_file = traj_folder_path.joinpath(f'{traj_name}.avi')
    if rot_format == rot_formats[1]:
        vid_new_file = traj_folder_path.joinpath(f'{traj_name}_pose_euler.avi')
    else:
        vid_new_file = traj_folder_path.joinpath(f'{traj_name}_pose.avi')

    # Load original video
    vid_og = cv2.VideoCapture(vid_og_file.__str__())
    readable, img = vid_og.read()
    vid_height, vid_width, vid_channels = img.shape
    fps = vid_og.get(cv2.CAP_PROP_FPS)

    # Create video writer for modified video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vid_new = cv2.VideoWriter(vid_new_file.__str__(), fourcc, fps, (vid_width, vid_height))

    for frame_id in tqdm(range(len(frame_pose))):
        if not readable:
            break
        # Overlay pose
        pose = frame_pose.iloc[frame_id]
        if rot_format == rot_formats[1]:
            img = overlay_pose_euler(img, pose[1:])
        else:
            img = overlay_pose_quaternions(img, pose[1:])

        # Write modified frame to new video
        vid_new.write(img)

        # Go to next frame
        readable, img = vid_og.read()

    vid_new.release()

    print(f"Modified video saved at {vid_new_file}")

