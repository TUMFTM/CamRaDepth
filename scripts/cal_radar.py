'''
   Generate radar inputs.
'''

import os
from os.path import join
import numpy as np
import argparse
from timeit import default_timer as timer

import torch
from nuscenes.nuscenes import NuScenes

import _init_paths
from fuse_radar import merge_selected_radar, cal_depthMap_flow, radarFlow2uv
from helper import create_tqdm_bar

if __name__ == '__main__':    
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
   
    nusc = NuScenes(args.version, dataroot = dir_nuscenes, verbose=False)
    dir_data_out = join(args.dir_data, 'prepared_data')   
        
    sample_indices = torch.load(join(args.dir_data,'data_split.tar'))['all_indices']
       
    N_total = len(sample_indices)
    print('Total sample number:', N_total)
        
    if start_idx == None:
        start_idx = 0
        
    if end_idx == None or end_idx > N_total - 1:
        end_idx = N_total -1
    
    frm_range = [0,4]
    
    loop = create_tqdm_bar(sample_indices[start_idx: end_idx+1])
    for ct, sample_idx in loop:
        
        start = timer()
        matrix = np.load(join(dir_data_out, '%05d_matrix.npz' % sample_idx))
        K = matrix['K']
        
        x1, y1, depth1, all_times1, x2, y2, depth2, all_times2, rcs, v_comp= merge_selected_radar(nusc, sample_idx, frm_range)
                
        depth_map1, flow, time_map1, rcs_map1, v_comp_map1 = cal_depthMap_flow(x1, y1, depth1, all_times1, x2, y2, depth2, all_times2, rcs, v_comp, downsample_scale=2, y_cutoff=34)        
        uv2 = radarFlow2uv(flow, K, depth_map1, downsample_scale=2, y_cutoff=34)
        
        radar_data = np.concatenate((depth_map1[..., None], uv2), axis=2) 
        
        np.save(join(dir_data_out, '%05d_radar.npy' % sample_idx), radar_data) 
         
        end = timer()
        t = end-start     
        loop.set_postfix(Time_used= '%.1f s' % t)
        
        