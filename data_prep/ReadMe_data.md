# Dataset creation
## Individual trajectory
Save all necessary files in a folder titled `[trajectory name]`:
   * Left camera video: `[trajectory name]_l.avi`
   * Right camera video: `[trajectory name]_r.avi`
   * Side camera video: `[trajectory name]_s.avi`
   * Camera time stamps: `[time_stamps].csv`
   * Robot pose logs: `[file name].txt`

## Dataset generation
1. Run the `get_pose_for_frames.py` file to match the robot positions to the camera frames
2. Run the `prep_dataset.py` file to convert the video files as images
    * Left, Right and Side camera videos will be saved as images in their respective `Left`, `Right`, and `Side` folders
    * Each image will have the same file name base: `[trajectory name]_[frame #]`
    * A `_l`, `_r`, or `_s` will be added after the base to complete the image file name
    * A new `[trajectory name].csv` annotation file will also be created and stored in the root dataset directory
    * The index of the annotation file will be the same as the image base name

## Video formatting
Run the `overlay_pose.py` script to overlay the robot pose on the merged video.