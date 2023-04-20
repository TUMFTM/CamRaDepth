from collections import namedtuple
import os, sys
import shutil
from pathlib import Path
import time
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..',))
import torch
import cv2
from data.dataloader import make_dataloaders
from utils.args import args
import numpy as np
from models.CamRaDepth import CamRaDepth


def load_without_module_state_dict(model, state_dict):
    """
    Load state dict without the 'module' prefix
    """
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    

if __name__ == "__main__":
    """
    Visualize the GT depth maps of the nuscenes dataset.
    """
    
    ##################### Data #####################
    num_samples = 50 # How many samples would you like to visualize?
    dataloaders = make_dataloaders(batchsize=1, split="test")
    test_dl = dataloaders["test"]
    dataloaders = make_dataloaders(batchsize=1, split="train")
    trai_dl, val_dl = dataloaders["train"], dataloaders["val"]
    splits = {"train": trai_dl, "val": val_dl, "test": test_dl}
    keys, values = list(splits.keys())[::-1], list(splits.values())[::-1]
    
    ##################### Create an output folder #####################
    path = Path(args.output_dir) / "visualzisations" # Where should it be created?
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    
    
    ##################### Load the model #####################
    test_name = "sunny"
    
    # If your GPU is currently occupied, you could simply use the CPU to visuazlie your recent predictions.
    use_gpu = True
    if use_gpu:
        cuda_id = 0
        with torch.cuda.device(cuda_id):
            model  = CamRaDepth(input_channels=args.input_channels)
            device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
            state = torch.load(args.checkpoint, map_location=device)
            load_without_module_state_dict(model, state['state_dict'])
            print(f"{args.hashtags_prefix} Loaded model from {args.checkpoint}")
            model.eval()
            model.to(device)
    else:
        model  = CamRaDepth(input_channels=args.input_channels)
        device = torch.device('cpu')
        state = torch.load(args.checkpoint, map_location=device)
        load_without_module_state_dict(model, state['state_dict'])
        print(f"{args.hashtags_prefix} Loaded model from {args.checkpoint}")
        model.eval()
        model.to(device)
        
    
    ##################### Visualize #####################
    print(f"{args.hashtags_prefix} Visualizing {num_samples} samples")
    for j, curr_dataloader in enumerate(values):
        
        cur_split_path = path / test_name / keys[j]
        orig_path = cur_split_path / "orig"
        depth_path = cur_split_path / "depth"
        seg_path = cur_split_path / "seg"
        gt_path = cur_split_path / "gt"
        radar_on_rgb_path = cur_split_path / "radar_on_rgb"
        radar_path = cur_split_path / "radar"
        seg_pred_path = cur_split_path / "seg_pred"
        collage_path = cur_split_path / "collage"
        depth_pred_path = cur_split_path / "depth_pred"
       
        os.makedirs(radar_path, exist_ok=True)
        os.makedirs(collage_path, exist_ok=True)
        
        loop = tqdm(enumerate(curr_dataloader), total=num_samples, desc=f"Processing {keys[j]}")
        for i, batch in loop:
            name = batch["name"][0].split(".")[0]
            curr_img_path = cur_split_path / name
            os.makedirs(curr_img_path, exist_ok=True)
            from matplotlib import pyplot as plt
            img = batch["image"][:, :3]
            
            # Orig image
            
            orig_img = batch["orig_img"].squeeze(0).cpu().numpy()
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(curr_img_path / "orig.png"), orig_img)

            # Lidar Groundtruth:
            lidar_gt = batch["gt"]["depth"]["lidar_depth"].squeeze().cpu().numpy()
            plt.imsave(str(curr_img_path / "lidar_gt.png"), lidar_gt, cmap="jet")
            gt_img = orig_img.copy()
            gt_colour = np.array(cv2.imread(str(curr_img_path / "lidar_gt.png")))
            gt_colour = cv2.cvtColor(gt_colour, cv2.COLOR_BGR2RGB)
            gt_img[lidar_gt > 0] = gt_colour[lidar_gt > 0]
            plt.imsave(str(curr_img_path / "lidar_gt.png"), gt_img)
            
            # Segmentation
            seg = batch["gt"]["seg"]["final_seg"].cpu().to(torch.long).permute(1, 2, 0).cpu().numpy().squeeze()
            plt.imsave(str(curr_img_path / "seg.png"), seg, cmap="rainbow")
            
            # pred_Seg
            with torch.no_grad():
                pred = model(batch["image"].to(torch.float32).to(device)[:, :args.input_channels])
            pred_seg = pred["seg"]["final_seg"]
            if pred_seg is not None:
                pred_seg = torch.max(pred_seg, dim=1)[1].detach().cpu().numpy().squeeze()
                plt.imsave(str(curr_img_path / "pred_seg.png"), pred_seg, cmap="rainbow")

            # TODO: Depth prediction
            depth_pred = pred["depth"]["final_depth"]
            depth_pred = depth_pred.detach().cpu().numpy().squeeze()
            plt.imsave(str(curr_img_path / "depth_pred.png"),  depth_pred, cmap="jet")
            
            
            # Radar
            radar_img = orig_img.copy()
            radar_img = cv2.cvtColor(radar_img, cv2.COLOR_RGB2GRAY)
            radar_img = np.repeat(radar_img[:, :, None], 3, axis=2)
            
            radar_data = batch["image"][:, 3] # .detach().cpu().numpy().squeeze()
            radar_data[radar_data != 0] = 1 - radar_data[radar_data != 0]
            radar_data = radar_data.squeeze().detach().cpu().numpy()
            radar_data = cv2.dilate(radar_data, np.ones((5, 5), np.uint8), iterations=1)
            plt.imsave(str(radar_path / f"{keys[j]}_{batch['name'][0]}"), radar_data, cmap="jet")
            radar_colour = np.array(cv2.imread(str(radar_path / f"{keys[j]}_{batch['name'][0]}")))
            radar_colour = cv2.cvtColor(radar_colour, cv2.COLOR_BGR2RGB)
            radar_img[radar_data > 0] = radar_colour[radar_data > 0]
            plt.imsave(str(curr_img_path / "radar.png"), radar_img)
            

            # Transperent colormap on the rgb.
            depth_color = cv2.imread(str(curr_img_path / "depth_pred.png"))
            depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
            
            # Blend the colorized depth map with the RGB image
            blended = cv2.addWeighted(orig_img, 0.8, depth_color, 0.75, 0)
            plt.imsave(str(curr_img_path / "depth_on_rgb.png"), blended)
            
            fig, axs = plt.subplots(2, 3, figsize=(20, 20))

            # Display the first image in the top left subplot
            axs[0, 0].imshow(orig_img)

            # Display the second image in the top right subplot
            axs[0, 1].imshow(seg, cmap="jet")
            
            if pred_seg is not None:
                axs[0, 2].imshow(pred_seg, cmap="rainbow")

            # Display the third image in the bottom left subplot
            axs[1, 0].imshow(depth_pred, cmap="jet")

            # Display the fourth image in the bottom right subplot
            if pred["seg"]["unsup_map"] is not None:
                pred_seg = pred["seg"]["unsup_map"].detach().cpu().numpy().squeeze()
                plt.imsave(str(curr_img_path / "unsup.png"), pred_seg)
                axs[1, 2].imshow(pred_seg, cmap="rainbow")
            else:
                axs[1, 2].imshow(blended)
            
            axs[1, 1].imshow(gt_img, cmap="jet")

            # Save the figure
            plt.savefig(str(collage_path / f"{keys[j]}_{batch['name'][0]}"))
                        
            plt.close()
            if i == num_samples:
                exit()
                


