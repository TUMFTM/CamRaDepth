import numpy as np
import glob
import os

from numpy import recarray
import imageio
import matplotlib.pyplot as plt

classes = [ 'backpack', 'umbrella', 'bag', 'tie', 'suitcase', 'case', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'animal_other', 'microwave', 'radiator', 'oven', 'toaster', 'storage_tank', 'conveyor_belt', 'sink', 'refrigerator', 'washer_dryer', 'fan',
            'dishwasher', 'toilet', 'bathtub', 'shower', 'tunnel', 'bridge', 'pier_wharf', 'tent', 'building', 'ceiling', 'laptop', 'keyboard', 'mouse', 
            'remote', 'cell phone', 'television', 'floor', 'stage', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 
            'donut', 'cake', 'fruit_other', 'food_other', 'chair_other', 'armchair', 'swivel_chair', 'stool', 'seat', 'couch', 'trash_can', 'potted_plant', 
            'nightstand', 'bed', 'table', 'pool_table', 'barrel', 'desk', 'ottoman', 'wardrobe', 'crib', 'basket', 'chest_of_drawers', 'bookshelf', 
            'counter_other', 'bathroom_counter', 'kitchen_island', 'door', 'light_other', 'lamp', 'sconce', 'chandelier', 'mirror', 'whiteboard', 'shelf', 
            'stairs', 'escalator', 'cabinet', 'fireplace', 'stove', 'arcade_machine', 'gravel', 'platform', 'playingfield', 'railroad', 'road', 'snow', 
            'sidewalk_pavement', 'runway', 'terrain', 'book', 'box', 'clock', 'vase', 'scissors', 'plaything_other', 'teddy_bear', 'hair_dryer', 'toothbrush', 
            'painting', 'poster', 'bulletin_board', 'bottle', 'cup', 'wine_glass', 'knife', 'fork', 'spoon', 'bowl', 'tray', 'range_hood', 'plate', 'person', 
            'rider_other', 'bicyclist', 'motorcyclist', 'paper', 'streetlight', 'road_barrier', 'mailbox', 'cctv_camera', 'junction_box', 'traffic_sign', 
            'traffic_light', 'fire_hydrant', 'parking_meter', 'bench', 'bike_rack', 'billboard', 'sky', 'pole', 'fence', 'railing_banister', 'guard_rail', 
            'mountain_hill', 'rock', 'frisbee', 'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard', 'surfboard', 
            'tennis_racket', 'net', 'base', 'sculpture', 'column', 'fountain', 'awning', 'apparel', 'banner', 'flag', 'blanket', 'curtain_other', 
            'shower_curtain', 'pillow', 'towel', 'rug_floormat', 'vegetation', 'bicycle', 'car', 'autorickshaw', 'motorcycle', 'airplane', 'bus', 'train', 
            'truck', 'trailer', 'boat_ship', 'slow_wheeled_object', 'river_lake', 'sea', 'water_other', 'swimming_pool', 'waterfall', 'wall', 'window', 
            'window_blind']

my_classes = [  'cat', 'dog', 'animal_other', 'tunnel', 'bridge', 'pier_wharf', 'tent', 'building', 'gravel', 'platform', 'playingfield', 'railroad', 'road', 
                'snow', 'sidewalk_pavement', 'runway', 'terrain', 'person', 'rider_other', 'bicyclist', 'motorcyclist', 'paper', 'streetlight', 'road_barrier', 
                'mailbox', 'cctv_camera', 'junction_box', 'traffic_sign', 'traffic_light', 'fire_hydrant', 'parking_meter', 'bench', 'bike_rack', 'billboard', 
                'sky', 'pole', 'fence', 'railing_banister', 'guard_rail', 'mountain_hill', 'rock', 'vegetation', 'bicycle', 'car', 'autorickshaw', 
                'motorcycle', 'airplane', 'bus', 'truck', 'train', 'trailer', 'boat_ship', 'slow_wheeled_object', 'wall', 'window', 'window_blind']

allocation_reduced_classes = {  'animal': ['cat', 'dog', 'animal_other'],
                                'tunnel': ['tunnel'],
                                'bridge': ['bridge'],
                                'building': ['building','platform'],
                                'road': ['road'],
                                'no_drive_road': ['sidewalk_pavement', 'railroad', 'runway'],
                                'terrain': ['terrain', 'playingfield'],
                                'person': ['person', 'rider_other', 'bicyclist', 'motorcyclist'],
                                'pole': ['pole', 'streetlight'],
                                'roadbarrier': ['road_barrier'],
                                'side_object': ['mailbox', 'junction_box', 'fire_hydrant', 'parking_meter'],
                                'traffic_sign': ['traffic_sign', 'billboard'],
                                'traffic_light': ['traffic_light'],
                                'bench': ['bench', 'bike_rack'],
                                'sky': ['sky'],
                                'fence': ['fence', 'railing_banister', 'guard_rail'],
                                'vegetation': ['vegetation', 'mountain_hill', 'rock'],
                                'two_wheels': ['bicycle', 'motorcycle', 'slow_wheeled_object'],
                                'car': ['car', 'autorickshaw'],
                                'truck': ['bus', 'truck', 'trailer'],
                                'plane_surface': ['wall', 'window']}

this_dir = os.path.dirname(__file__)
dir_data = os.path.join(this_dir, '../../nuscenes_mini')

def vehicle_seg_only(idx = None, save = False):

    if idx == None:   
        list = np.array(glob.glob("../external/mseg/mseg-semantic/temp_files/mseg-3m_mseg_test_universal_ms/360/gray/*_im.png"))
        list = np.sort(list)
    else:
        list = np.array(glob.glob("../external/mseg/mseg-semantic/temp_files/mseg-3m_mseg_test_universal_ms/360/gray/" + str(idx) + "_im.png"))


    for im_path in list:
        im = imageio.imread(im_path)
        # filter car = 176, bus = 180, truck = 182
        veh_seg = np.logical_or(im==176, im==180, im==182)
        if save: 
            file_name = im_path[-17:]
            sample_name = file_name[:-7]
            path_seg = dir_data + "/prepared_data/" + sample_name + "_mseg.npy"
            print(path_seg)
            os.makedirs(path_seg, exist_ok=True)
            np.save(path_seg, veh_seg)
        else:
            plt.imshow(veh_seg)
            plt.show()  
                 

def mseg(idx = None, save = False):

    if idx == None:
        list = np.array(glob.glob(os.path.join(this_dir, "../external/mseg/mseg-semantic/temp_files/mseg-3m_prepared_data_universal_ms/360/gray/*_im.png")))
        list = np.sort(list)
    else:
        list = np.array(glob.glob(os.path.join(this_dir, "../external/mseg/mseg-semantic/temp_files/mseg-3m_prepared_data_universal_ms/360/gray/" + str(idx) + "_im.png")))
        
    for im_path in list:
        im_array = imageio.imread(im_path)
        # label everything 0 that is not in classes
        im_array[~np.isin(im_array, np.array(my_numbers))] = 0
        for j in range(len(my_numbers)):
            im_array[im_array == my_numbers[j]] = new_numbers[j]

        if save:
            file_name = im_path[-17:]
            sample_name = file_name[:-7]
            path_seg = dir_data + "/prepared_data/" + sample_name + "_mseg.npy"
            print("Saving File: " + path_seg)
            np.save(path_seg, im_array)
                        
        else:
            plt.imshow(im_array)
            plt.show()

def reduced_mseg(idx = None, save = False):

    if idx == None:   
        # list = np.array(glob.glob(dir_data + "/prepared_data/*_rain_mseg.npy"))
        list = np.array(glob.glob(dir_data + "/prepared_data/*_mseg.npy"))
        list = np.sort(list)
    else:
        # list = np.array(glob.glob(dir_data + "/prepared_data/" + str(idx) + "_rain_mseg.npy"))
        list = np.array(glob.glob(dir_data + "/prepared_data/" + str(idx) + "_mseg.npy"))

        
    for npy_path in list:
        im_array = np.load(npy_path)
        # label everything 0 that is not in classes
        im_array[~np.isin(im_array, np.array(reduced_numbers))] = 255
        new_number = 0
        for key in reduced_classes:
            for i in reduced_classes[key]:
                im_array[im_array == i] = new_number
            new_number +=1
           
        if save:
            file_name = npy_path[-14:]
            sample_name= file_name[:-9]
            path_seg = dir_data + "/prepared_data/" + sample_name + "_mseg.npy"
            print("Saving File: " + path_seg)
            # os.makedirs(path_seg, exist_ok=True)
            np.save(path_seg, im_array)
        else:
            plt.imshow(im_array)
            plt.show()
            

if __name__ == "__main__":
    
    numbers = list(range(len(classes)))
    all_class_numbers = dict(zip(classes, numbers))
    # filter for new classes and asign new label numbers
    my_class_numbers = {i: all_class_numbers[i] for i in my_classes}
    my_numbers = list(my_class_numbers.values())
    new_numbers = list(range(1,len(my_numbers)+1))
    new_class_numbers = {my_classes[i]: new_numbers[i] for i in range(len(my_classes))}
    print(new_class_numbers)

    # merge some of the new classes to reduce number of classes
    reduced_classes = allocation_reduced_classes.copy()
    reduced_numbers = []
    list_reduced_classes  = []
    for key in allocation_reduced_classes:
        reduced_classes[key] = []
        for i in allocation_reduced_classes[key]:
            reduced_classes[key].append(new_class_numbers[i])
            reduced_numbers.append(new_class_numbers[i])
            list_reduced_classes.append(i)
    
    #check which classes from my_classes are not used anymore in reduced_classes
    s = set(list_reduced_classes )
    diff = [x for x in my_classes if x not in s]
    print(diff)

    # mseg(save=True)
    reduced_mseg(save=True)