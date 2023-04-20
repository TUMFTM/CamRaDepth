import os
from os.path import join
from turtle import down
import numpy as np
import matplotlib.pyplot as plt
from numpy import outer
import skimage.io as io
from skimage.transform import resize
import argparse
from timeit import default_timer as timer

import torch
from nuscenes.nuscenes import NuScenes
from helper import create_tqdm_bar

import _init_paths
from fuse_radar import merge_selected_radar, cal_depthMap_flow, radarFlow2uv

def downsample_im(im, downsample_scale, y_cutoff):
    h_im, w_im = im.shape[0:2]        
    h_im = int( h_im / downsample_scale )
    w_im = int( w_im / downsample_scale ) 
    
    im = resize(im, (h_im,w_im,3), order=1, preserve_range=True, anti_aliasing=False) 
    im = im.astype('uint8')
    im = im[y_cutoff:,...]
    return im

def show(v_comp_map, im):
        
    h,w = im.shape[0:2]    
    x_map, y_map = np.meshgrid(np.arange(w), np.arange(h))
    
    # Plot input image
    fig, axs = plt.subplots(2,2)

    norm = plt.Normalize(0, v_comp_map.max())

    axs[0,0].imshow(im)
    msk = v_comp_map > 0
    axs[0,0].scatter(x_map[msk], y_map[msk], c=v_comp_map[msk], s=2, cmap='jet', norm=norm)
    axs[0,0].set_title("Radial Velocity")

    norm = plt.Normalize(0, 100)
    axs[0,1].imshow(im)
    msk = depth_map1 > 0
    axs[0,1].scatter(x_map[msk], y_map[msk], c=depth_map1[msk], s=2, cmap='jet', norm=norm)
    axs[0,1].set_title("Radar")

    axs[1,0].imshow(im)
    msk = (depth_map1 > 0) & (v_comp_map == 0)
    axs[1,0].scatter(x_map[msk], y_map[msk], c=depth_map1[msk], s=2, cmap='jet', norm=norm)
    axs[1,0].set_title("Filtered Radar")
    plt.show()

if __name__=='__main__':

    save = True
    show = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_data', type=str)
    parser.add_argument('--version', type=str, default='v1.0-mini')
    parser.add_argument('--start_idx', type=int)
    parser.add_argument('--end_idx', type=int)
    
    args = parser.parse_args()  

    if args.dir_data == None:
        this_dir = os.path.dirname(__file__)
        args.dir_data = os.path.join(this_dir, '../../nuscenes_mini/v1.0-mini')
    dir_nuscenes = os.path.join(args.dir_data)
    start_idx = args.start_idx
    end_idx = args.end_idx
        
    downsample_scale = 2
    y_cutoff = 34
   
    # sample_indices = torch.load(join(args.dir_data,'data_split.tar'))['rain_night_sample_indices']
    sample_indices = torch.load(join(args.dir_data,'data_split.tar'))['all_indices']
    N_total = len(sample_indices)

    nusc = NuScenes(args.version, dataroot = dir_nuscenes, verbose=False)

    if end_idx == None or end_idx > N_total - 1:
        end_idx = N_total -1
    
    #frame 0-4 = 5 sweeps
    frm_range = [0,4]
    
    loop = create_tqdm_bar(sample_indices[start_idx: end_idx+1])
    for ct, sample_idx in loop:

        cam_token = nusc.sample[sample_idx]['data']['CAM_FRONT']
        cam_data = nusc.get('sample_data', cam_token)
        cam_token2 = cam_data['next']                    
        cam_data2 = nusc.get('sample_data', cam_token2)
        cam_path2 = join(nusc.dataroot, cam_data2['filename'])
        im2 = io.imread(cam_path2)
        im = downsample_im(im2, downsample_scale=2, y_cutoff=34)
        
        x1, y1, depth1, all_times1, x2, y2, depth2, all_times2, rcs, v_comp= merge_selected_radar(nusc, sample_idx, frm_range)
                    
        depth_map1, flow, time_map1, rcs_map1, v_comp_map1 = cal_depthMap_flow( x1, y1, depth1, all_times1, x2, y2, depth2, all_times2,
                                                                                    rcs, v_comp, downsample_scale=2, y_cutoff=34)
        
        if save:
            dir_data_out = os.path.join(args.dir_data, 'prepared_data') 
            # np.save(join(dir_data_out, '%05d_rain_rad_vel.npy' % sample_idx), v_comp_map1)
            np.save(join(dir_data_out, '%05d_rad_vel.npy' % sample_idx), v_comp_map1)         
            loop.set_postfix({"saved": True})

        if show:
            show(v_comp_map1, im)