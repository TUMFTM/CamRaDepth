'''
Extract images

'''

import skimage.io as io
import os
from os.path import join
import glob
import argparse
from skimage.transform import resize
import torch
import numpy as np

from nuscenes.nuscenes import NuScenes
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_data', type=str)
    parser.add_argument('--version', type=str, default='v1.0-trainval')
        
    args = parser.parse_args()

    if args.dir_data == None:
        this_dir = os.path.dirname(__file__)
        args.dir_data = os.path.join(this_dir, '..', 'data')
    dir_nuscenes = os.path.join(args.dir_data, 'nuscenes/raw')

       
    nusc = NuScenes(args.version, dataroot = dir_nuscenes, verbose=False)
    
    dir_data_out = "/home/ubuntu/lukas/depthcompletion/Radar_Depth_Completion/external/images_original_size/"
    sample_indices = torch.load(join(args.dir_data,'data_split.tar'))['all_indices'] 
         
    ct = 0         
    for sample_idx in sample_indices:

        cam_token = nusc.sample[sample_idx]['data']['CAM_FRONT']
        cam_data = nusc.get('sample_data', cam_token)
        
        if cam_data['next']:
                
            cam_token2 = cam_data['next']                    
            cam_data2 = nusc.get('sample_data', cam_token2)
            cam_path2 = join(nusc.dataroot, cam_data2['filename'])
            im = io.imread(cam_path2)
         
                       
            io.imsave(join(dir_data_out, '%05d_im.png' % sample_idx), im)         
           
        ct += 1
        print('Save image %d/%d' % ( ct, len(sample_indices) ) )
        

    
    