import os
import numpy as np
import time
import argparse
import glob
from pathlib import Path
 
def save_list(img_filename_list, radar_filename_list, filt_radar_filename_list, mseg_filename_list,
                im_uv_filename_list, rad_vel_filename_list, gt_filename_list, file_name="new_split", split="train"):
    

    des_length = len(img_filename_list)
    assert len(gt_filename_list) >= des_length and len(radar_filename_list) >= des_length and len(filt_radar_filename_list) >= des_length  \
        and len(mseg_filename_list) >= des_length and len(im_uv_filename_list) >= des_length and len(rad_vel_filename_list) >= des_length, f"Length is {des_length}"

    if len(img_filename_list) == 0 or len(gt_filename_list) == 0:
        print("List(s) empty!")
        print("Number of images: {} \nNumber of ground truths: {} \n".format(len(img_filename_list), len(gt_filename_list)))
    else:
        if len(img_filename_list) == len(gt_filename_list):
            list_with_all_files = list(zip( img_filename_list,
                                            radar_filename_list,
                                            filt_radar_filename_list,
                                            mseg_filename_list,
                                            im_uv_filename_list,
                                            rad_vel_filename_list,
                                            gt_filename_list))
            
            path = Path("../CamRaDepth/data")
            os.makedirs(path, exist_ok=True)
            path = path / file_name
            np.save(path, list_with_all_files)
            print("Saved split file: " + str(path))

        else:
            print("Number of images {} does not match number of ground truths {}!".format(len(img_filename_list), len(gt_filename_list)))


def create_file_list(dir_data, file_name="new_split"):
    """
    Organize the raw data as needed, and save the created list to dist
    so later on we could simply load the the prepared data, instead of
    reorganizing each time, for a significant speed up (0.1 seconds vs 29 seconds)
    """
    start = time.time()
    dir_data = os.path.join(dir_data, "prepared_data/")
    print(dir_data)
    img_filename_list = glob.glob(dir_data + '*_im.jpg')
    img_filename_list.sort()
    radar_filename_list = glob.glob(dir_data + '*_radar.npy')
    radar_filename_list.sort()
    filt_radar_filename_list = glob.glob(dir_data + '*_radar_filtered.npy')
    filt_radar_filename_list.sort()
    mseg_filename_list = glob.glob(dir_data + '*_mseg.npy')
    mseg_filename_list.sort()
    im_uv_filename_list = glob.glob(dir_data + '*_im_uv.npy')
    im_uv_filename_list.sort()
    rad_vel_filename_list = glob.glob(dir_data + '*_rad_vel.npy')
    rad_vel_filename_list.sort()
    gt_filename_list = glob.glob(dir_data + '*_gt.npy')
    gt_filename_list.sort()
    save_list(img_filename_list, radar_filename_list, filt_radar_filename_list, mseg_filename_list,
                im_uv_filename_list, rad_vel_filename_list, gt_filename_list, file_name=file_name)
    end = time.time()
    print("Duration: ", end - start, " seconds")
    
    
if __name__ == '__main__':
    
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--dir_data', type=str, default='data/prepard_data', help='path to the data')
    argparse.add_argument('--file_name', type=str, default='new_split', help='name of the file')
    args = argparse.parse_args()
    
    create_file_list(args.dir_data, args.file_name)
    