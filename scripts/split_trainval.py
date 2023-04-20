'''
Split data with moving ego vehicles on a clear day.

'''

import os
import numpy as np
import torch
import argparse

from nuscenes.nuscenes import NuScenes

np.random.seed(1)

def is_first_2_sample_in_scene(idx):
        if not nusc.sample[idx]['prev']:
            return True
        elif not nusc.sample[idx-1]['prev']:
            return True
        else:
            return False
        
def is_last_2_sample_in_scene(idx):
    if not nusc.sample[idx]['next']:
        return True
    elif not nusc.sample[idx+1]['next']:
        return True
    else:
        return False
    
def cal_moving_forward_distance(idx):    
    pose = nusc.get('ego_pose', nusc.get('sample_data', nusc.sample[idx]['data']['LIDAR_TOP'])['ego_pose_token'])['translation']
    pose_next = nusc.get('ego_pose', nusc.get('sample_data', nusc.sample[idx+1]['data']['LIDAR_TOP'])['ego_pose_token'])['translation']        
    distance = ( (pose[0] - pose_next[0])**2 + (pose[1] - pose_next[1])**2 ) ** 0.5    
    return distance


def stop_in_neighboring_4_samples(sample_idx, thres=0.1):
    dist_b2 = cal_moving_forward_distance(sample_idx-2)
    dist_b1 = cal_moving_forward_distance(sample_idx-1)
    dist_f1 = cal_moving_forward_distance(sample_idx)
    dist_f2 = cal_moving_forward_distance(sample_idx+1)
 
    if dist_b1 < thres or dist_b2 < thres or dist_f1 < thres or dist_f2 < thres:
        return True
    else:
        return False
    
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_data', type=str)
    parser.add_argument('--version', type=str, default='v1.0-mini')
        
    args = parser.parse_args()

    if args.dir_data == None:
        this_dir = os.path.dirname(__file__)
        args.dir_data = os.path.join(this_dir, '../../nuscenes_mini/v1.0-mini')
    dir_nuscenes = os.path.join(args.dir_data) # , 'nuscenes/raw')


    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
                
    nusc = NuScenes(version=args.version, dataroot = dir_nuscenes, verbose=False)
    n_step = 1
        
    clear_day_moving_scenes = []
    rain_night_moving_scenes = []

    for scene in nusc.scene:
        if 'wait'.lower() in scene['description'].lower():
            continue   
        if 'rain'.lower() in scene['description'].lower() or 'Night'.lower() in scene['description'].lower():
            rain_night_moving_scenes.append(scene['token'])
        else:
            clear_day_moving_scenes.append(scene['token'])
        
    np.random.shuffle(clear_day_moving_scenes)
    np.random.shuffle(rain_night_moving_scenes)
     
    print(len(nusc.scene), 'total scenes')
    print(len(clear_day_moving_scenes), 'clear_day_moving_scenes')
    print(len(rain_night_moving_scenes), 'rain_night_moving_scenes')
    
    
    # n_train_scenes = int(round(len(clear_day_moving_scenes) * train_ratio))
    # n_val_scenes = int(round(len(clear_day_moving_scenes) * val_ratio))
    # n_test_scenes = len(clear_day_moving_scenes) - n_train_scenes - n_val_scenes
    
    n_train_scenes = int(round(len(nusc.scene) * train_ratio))
    n_val_scenes = int(round(len(nusc.scene) * val_ratio))
    n_test_scenes = len(nusc.scene) - n_train_scenes - n_val_scenes
        
    if args.version == 'v1.0-mini':
        n_train_scenes = 2
        n_val_scenes = 1
        n_test_scenes = 1
    
    train_scenes = []
    val_scenes = []
    test_scenes = []
    
    scene = clear_day_moving_scenes
    train_scenes += scene[:n_train_scenes]
    val_scenes += scene[ n_train_scenes: n_train_scenes + n_val_scenes ]
    test_scenes += scene[n_train_scenes + n_val_scenes : ]
    
    train_scenes = train_scenes[::n_step]
    val_scenes = val_scenes[::n_step]
    test_scenes = test_scenes[::n_step]
       
    train_sample_idx = []
    val_sample_idx = []
    test_sample_idx = []
    
    
    ############## Split train, val and test
    for idx, sample in enumerate(nusc.sample):        
        if is_first_2_sample_in_scene(idx) or is_last_2_sample_in_scene(idx) or stop_in_neighboring_4_samples(idx):
            continue        
        if sample['scene_token'] in train_scenes:
            train_sample_idx.append(idx)
        elif sample['scene_token'] in val_scenes:
            val_sample_idx.append(idx)
        elif sample['scene_token'] in test_scenes:
            test_sample_idx.append(idx)
            
    if 26198 in train_sample_idx:
        train_sample_idx.remove(26198)
    elif 26198 in val_sample_idx:
        val_sample_idx.remove(26198)
    elif 26198 in test_sample_idx:
        test_sample_idx.remove(26198)

    all_idx = train_sample_idx + val_sample_idx + test_sample_idx
    
    data_split = {'all_indices': all_idx,
                  'train_sample_indices': train_sample_idx,
                  'val_sample_indices': val_sample_idx,
                  'test_sample_indices':  test_sample_idx }
        
    print(len(train_sample_idx), len(val_sample_idx), len(test_sample_idx))
    
    
    ############## Split according to weather conditions
    # clear_day_sample_idx = []
    # rain_night_sample_idx = []
    # for idx, sample in enumerate(nusc.sample):  
    #     if is_first_2_sample_in_scene(idx) or is_last_2_sample_in_scene(idx) or stop_in_neighboring_4_samples(idx):
    #         continue        
    #     if sample['scene_token'] in clear_day_moving_scenes:
    #         clear_day_sample_idx.append(idx)
    #     elif sample['scene_token'] in rain_night_moving_scenes:
    #        rain_night_sample_idx.append(idx)
            
    
    # if 26198 in clear_day_sample_idx:
    #     clear_day_sample_idx.remove(26198)
    # elif 26198 in rain_night_sample_idx:
    #     rain_night_sample_idx.remove(26198)

    # print(len(clear_day_sample_idx), len(rain_night_sample_idx))
    
    # all_idx = clear_day_sample_idx + rain_night_sample_idx
    
    # data_split = {'all_indices': all_idx,
    #               'clear_day_sample_indices': clear_day_sample_idx,
    #               'rain_night_sample_indices': rain_night_sample_idx }
    # all_idx = clear_day_sample_idx + rain_night_sample_idx
    
    # data_split = {'all_indices': all_idx,
    #               'clear_day_sample_indices': clear_day_sample_idx,
    #               'rain_night_sample_indices': rain_night_sample_idx }
    

    torch.save(data_split, os.path.join(args.dir_data, 'data_split.tar'))