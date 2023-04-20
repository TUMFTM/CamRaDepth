#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

DATA_DIR=$SCRIPT_DIR"/../../nuscenes_mini"
DATA_VERSION="v1.0-mini"

cd $SCRIPT_DIR

echo -e "Preparing nuScenes" $DATA_VERSION "\n"

# 1) split data
echo "###############################"
echo "1) Splitting data:"
python split_trainval.py --version $DATA_VERSION --dir_data $DATA_DIR
echo -e "###############################\n"

# 2) extract images for flow computation
echo "###############################"
echo "2) preparing image flow extraction:"
python prepare_flow_im.py --version $DATA_VERSION --dir_data $DATA_DIR
echo -e "###############################\n"

# 3) compute image flow from im1 to im2
echo "###############################"
echo "3) Computing image flow:"
python cal_flow.py --dir_data $DATA_DIR
echo -e "###############################\n"

# 4) compute camera intrinsic matrix and transformation from cam1 to cam2
echo "###############################"
echo "4) Calculating camera calibration matrices:"
python cal_cam_matrix.py --version $DATA_VERSION --dir_data $DATA_DIR
echo -e "###############################\n"

# 5) transform image flow to normalized expression (u2,v2)
echo "###############################"
echo "5) Calculating image flow uv"
python cal_im_flow2uv.py --dir_data $DATA_DIR
echo -e "###############################\n"

# 6) compute vehicle semantic segmentation
echo "###############################"
echo "6) Segmenting vehicles:"
python semantic_seg.py --dir_data $DATA_DIR
echo -e "###############################\n"

# 7) compute dense ground truth (depth1, u2, v2) and low height mask
echo "###############################"
echo "7) Assembling ground truth depth data:"
python cal_gt.py --version $DATA_VERSION --dir_data $DATA_DIR
echo -e "###############################\n"

# 8) compute merged radar (5 frames)
echo "###############################"
echo "8) Merging radar reflections:"
python cal_radar.py --version $DATA_VERSION --dir_data $DATA_DIR
echo -e "###############################\n"

# 9) get radar velocity
echo "###############################"
echo "9) Extracting radar velocity:"
python show_v_comp.py --version $DATA_VERSION --dir_data $DATA_DIR
echo -e "###############################\n"

# 10) filter radar values
echo "###############################"
echo "10) Filtering radar data:"
python depth_difference.py --dir_data $DATA_DIR
echo -e "###############################\n"

# 11) Create a split.npy file
echo "###############################"
echo "11) Creating split file:"
python create_a_split_file.py --dir_data $DATA_DIR
echo -e "###############################\n"

# cd back to repository root
cd ..
echo "finished preprocessing data!"