from multiprocessing.sharedctypes import RawArray
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob
import os
import argparse

def sid_depth_thresh(input_depth):

    alpha = 5
    beta = 16
    K = 100

    depth_thresh = np.exp(((input_depth * np.log(beta / alpha)) / K) + np.log(alpha))
    depth_thresh = 100

    return depth_thresh

def filter_radar_by_lidar(radar, gt):
    
    radar = np.moveaxis(radar, -1, 0)
    # take only first channel of radar to get depth map and ignore uv
    radar_depth = radar[0, :, :]

    # take only first channel of gt to get depth map and ignore uv
    gt = gt[:, :, 0]

    diff = np.zeros(np.shape(radar_depth))
    msk_radar = radar_depth > 0
    diff[msk_radar] = gt[msk_radar] - radar_depth[msk_radar]

    filtered_radar = np.zeros(np.shape(radar_depth))
    msk_filtered= abs(radar_depth - gt) <= sid_depth_thresh(gt)
    filtered_radar[msk_filtered] = radar_depth[msk_filtered]

    return filtered_radar, diff


#image = np.array(Image.open(params.dir_data + params.idx + "_im.jpg"))
def save(dir_data):
    list_pairs = []
    radar_list = np.sort(np.array(glob.glob(dir_data + "/prepared_data/*_radar.npy")))
    gt_list = np.sort(np.array(glob.glob(dir_data + "/prepared_data/*_gt.npy")))
    if len(radar_list) == len(gt_list):
        list_pairs = list(zip(radar_list, gt_list))
    else:
        print("Error: Not same list length")

    ct = 0
    for idx in range(len(list_pairs)):

        radar_path = list_pairs[idx][0]
        radar = np.load(radar_path)
        gt_path = list_pairs[idx][1]
        gt = np.load(gt_path)

        filtered_radar, _ = filter_radar_by_lidar(radar, gt)

        file_name = gt_path[-12:]
        sample_idx = file_name[:-7]
        path_seg = dir_data + "/prepared_data/" + sample_idx + "_radar_filtered.npy"
        # print(path_seg)
        ct += 1
        print('Save Numpy Array %d/%d' % ( ct, len(radar_list) ) )
        np.save(path_seg, filtered_radar)

def show(dir_data, idx):
    # Creating 3 subplots with predicted depth, gt and errormap
    image = np.array(Image.open(dir_data + "/prepared_data/" + str(idx) + "_im.jpg"))
    radar = np.load(dir_data + "/prepared_data/" + str(idx) + "_radar.npy")
    gt = np.load(dir_data + "/prepared_data/" + str(idx) + "_gt.npy")

    filtered_radar, diff_depth = filter_radar_by_lidar(radar, gt)

    radar = np.moveaxis(radar, -1, 0)
    # take only first channel of radar to get depth map and ignore uv
    radar_depth = radar[0, :, :]
    gt = gt[:, :, 0]

    fig, axs = plt.subplots(2, 2)

    h,w = image.shape[0:2]    
    x_map, y_map = np.meshgrid(np.arange(w), np.arange(h))
        

    axs0 = axs[0, 0]
    axs0.imshow(image)
    msk = radar_depth > 0
    sc = axs0.scatter(x_map[msk], y_map[msk], c=radar_depth[msk], s=1, cmap='jet')
    axs[0, 0].set_title('Radar depth')
    
    # Plot GT Depth
    axs1 = axs[1, 0]
    axs1.imshow(image)
    msk = gt > 0
    axs1.scatter(x_map[msk], y_map[msk], c=gt[msk], s=1, cmap='jet')
    axs[1, 0].set_title('GT depth')

    axs2 = axs[0, 1]
    #axs2.imshow(diff_depth, cmap="brg")
    axs2.imshow(image)
    msk = diff_depth != 0
    axs2.scatter(x_map[msk], y_map[msk], c=abs(diff_depth[msk]), s=1, cmap='jet')
    axs[0, 1].set_title('Error_map')

    axs3 = axs[1, 1]
    axs3.imshow(image)
    msk = filtered_radar > 0
    axs3.scatter(x_map[msk], y_map[msk], c=filtered_radar[msk], s=1, cmap='jet')
    axs[1, 1].set_title('Filtered Radar depth')

    cbar_ax = fig.add_axes([0.85, 0.05, 0.05, 0.7])
    fig.colorbar(sc, cax = cbar_ax)
    plt.show()

if __name__ == "__main__":
    
    show_data = False
    save_data = True
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_data', type=str)
    
    args = parser.parse_args() 
    
    if args.dir_data == None:
        this_dir = os.path.dirname(__file__)
        args.dir_data = os.path.join(this_dir, '../../nuscenes_mini/v1.0-mini')
    dir_data = args.dir_data

    
    if show_data:
        idx_list = []
        idx_list = [item for item in input("Enter index of sample: ").split()]

        for idx in idx_list:
            print(dir_data, idx)
            show(dir_data, idx)
        
    if save_data:
        save(dir_data)
