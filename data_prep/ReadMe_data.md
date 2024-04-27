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

## Creating pre-determined train and test sets
The `split_train_test.py` script creates train and test sets from the annotation files of multiple trajectories. 
The all annotation files are merged, and then images are split based on their position in the cube of possible mandible positions. 
The annotation subsets of each subsection are then split into train and test sets, which are then combined to produce the final train and test annotation files. 

The test set is created by using `test_ratio` images from all available annotations. The remainer are then used to create the complete training set, of which `valid_ratio`% of will be used to create the validation set. Images left over after that will make up the true training set. 
