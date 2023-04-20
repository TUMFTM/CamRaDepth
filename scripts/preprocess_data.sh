#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

DATA_DIR=$SCRIPT_DIR"/../../nuscenes_mini"
DATA_VERSION="v1.0-mini"

cd $SCRIPT_DIR

echo "Preparing nuScenes" $DATA_VERSION

# 1) split data
echo "splitting data..."
python split_trainval.py --version $DATA_VERSION --dir_data $DATA_DIR

# 2) extract images for flow computation
echo "preparing image flow extraction..."
python prepare_flow_im.py --version $DATA_VERSION --dir_data $DATA_DIR

# 3) compute image flow from im1 to im2
echo "computing image flow..."
python cal_flow.py --dir_data $DATA_DIR

# 4) compute camera intrinsic matrix and transformation from cam1 to cam2
echo "calculating camera calibration matrices..."
python cal_cam_matrix.py --version $DATA_VERSION --dir_data $DATA_DIR

# 5) transform image flow to normalized expression (u2,v2)
echo "calculating image flow uv..."
python cal_im_flow2uv.py --dir_data $DATA_DIR

# 6) compute vehicle semantic segmentation
echo "segmenting vehicles..."
python semantic_seg.py --dir_data $DATA_DIR

# 7) compute dense ground truth (depth1, u2, v2) and low height mask
echo "assembling ground truth depth data..."
python cal_gt.py --version $DATA_VERSION --dir_data $DATA_DIR

# 8) compute merged radar (5 frames)
echo "merging radar reflections..."
python cal_radar.py --version $DATA_VERSION --dir_data $DATA_DIR

# 9) get radar velocity
echo "extracting radar velocity.."
python show_v_comp.py --version $DATA_VERSION --dir_data $DATA_DIR

# 10) filter radar values
echo "filtering radar data..."
python depth_difference.py --dir_data $DATA_DIR

# 11) Create a split.npy file
python create_a_split_file.py --dir_data $DATA_DIR

# cd back to repository root
cd ..
echo "finished preprocessing data!"