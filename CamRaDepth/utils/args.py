import os, sys
from pathlib import Path
import argparse
from easydict import EasyDict as edict
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add the main directory to the python path

this_dir = this_dir = os.path.dirname(__file__)

args = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Radar Depth Completion')

# Dataset
args.add_argument("--max_depth", type=float, default=100, help="Maximum depth in meters")
args.add_argument("--max_distances", type=list, default=[100, 50], help="Maximum distances in meters")
args.add_argument("--mini_dataset", action="store_true", help="If set, the mini dataset is used")
# args.add_argument("--train_val_split", type=list, default=[17902, 2237], help="Splits the Dataset in train and validation set; \
#     added it must be equal to num_samples")
args.add_argument("--image_dimension", type=tuple, default=(416, 800), help="Image dimension")
args.add_argument("--val_test_size", type=int, default=2237, help="Size of the validation and test set")
args.add_argument("--num_workers", type=int, default=8, help="Number of workers for the dataloaders")
args.add_argument("--split", type=str, default="original_split.npy", help="Path to the split file")

# Semantic Segmentation
args.add_argument("--supervised_seg", action="store_true", help="If set, we deploy a supervised semantic segmentation branch")
args.add_argument("--unsupervised_seg", action="store_true", help="If set, we deploy an unsupervised semantic segmentation branch")
args.add_argument("--num_classes", type=int, default=21, help="Number of classes in the semantic segmentation")

# Optimization
args.add_argument("--batch_size", type=int, default=2, help="Batch size")
args.add_argument("--desired_batch_size", type=int, default=None, help="Desired batch size")
args.add_argument("--num_epochs", type=int, default=30, help="Number of epochs")
args.add_argument("--num_steps", type=int, default=None, help="If set, the training session will run for a number of epochs that correspond \
                    to the number of steps and the batch size.")
args.add_argument("--learning_rate", type=float, default=6e-05, help="Learning rate")
args.add_argument("--early_stopping_thresh", type=int, default=10, help="Number of epochs to wait before early stopping")
args.add_argument("--groupnorm_divisor", type=int, default=16, help="Divisor for the group normalization")
args.add_argument("--cuda_id", type=int, default=0, help="Which GPU to use")
args.add_argument("--distributed", action="store_false", help="If set, nn.Dataparallel is used")

# Optimizer and scheduler
args.add_argument("--div_factor", type=float, default=2, help="Divisor for the OneCyclicLR scheduler")

# Model
args.add_argument("--input_channels", type=int, default=7, help="Number of input channels")
args.add_argument("--rgb_only", action="store_true", help="If set, only the RGB channels are used as input")
args.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
args.add_argument("--model", type=str, default="base", help="Model to use")
args.add_argument("--load_ckpt", action="store_true", help="If set, the checkpoint is loaded")

# Hyperparameters tuning
args.add_argument("--random_search_dataset_size", type=int, default=500, help="Size of the dataset used for the random search")
args.add_argument("--random_search_num_trials", type=int, default=50, help="Number of trials for the random search")

# Output
args.add_argument("--output_dir", type=str, default="Output", help="Output directory")
args.add_argument("--save_model", action="store_true", help="If set, the checkpoint and tensorboard logs are saved")
args.add_argument("--arch_name", type=str, default="Transformer", help="Name of the architecture")
args.add_argument("--run_name", type=str, default='current', help="Name of the run")
args.add_argument("--run_mode", type=str, default='train', help="Mode of the run")

# Visualization
args.add_argument("--num_vis", type=int, default=25, help="Number of samples to visualize")

args = args.parse_args()
args = edict(vars(args))


# print(args.load_ckpt);exit()


############################ Manual settings ############################

# Uncomment the following section, in order to manually set the hyperparameters and relevant paths (Recommended).
# The current settings are set as follows:
# 1. exp_index = 0: Base (only RGB)
# 2. exp_index = 1: Base (RGB + Radar)
# 3. exp_index = 2: Supervised semantic segmentation
# 4. exp_index = 3: Unsupervised semantic segmentation
# 5. exp_index = 4: Supervised + Unsupervised semantic segmentation
# 6. exp_index = 5: Supervised + Unsupervised semantic segmentation (Only RGB)



# exp_index = 1
# args.save_model = True
# args.load_ckpt = True
# args.distributed = False 
# args.run_mode = ["train", "test"][0]

           
# args.model = ["base (rgb)", "base", "supervised_seg", "unsupervised_seg", "sup_unsup_seg", "sup_unsup_seg (rgb)"][exp_index]

# checkpoints_path = os.path.join(this_dir, "../checkpoints")
# args.checkpoint = [os.path.join(checkpoints_path, "Base_RGB_TL.pth"),
#                        os.path.join(checkpoints_path, "Base_TL.pth"),
#                        os.path.join(checkpoints_path, "Seg_Sup_TL.pth"),
#                        os.path.join(checkpoints_path, "Seg_Unsup_TL.pth"),
#                        os.path.join(checkpoints_path, "Seg_Sup_Unsup_TL.pth"),
#                        os.path.join(checkpoints_path, "Seg_Sup_Unsup_RGB_FS.pth"),
#                         ][exp_index]

# args.arch_name = ["Debug", "Transformer"][0]  # 
# args.run_name = ["Base_RGB", "Base", "Seg_Supervised", "Seg_Unsupervised", "Seg_Sup_Unsup", "Seg_Sup_Unsup_RGB"][exp_index]
# args.batch_size = [2, 2, 2, 2, 2, 2][exp_index]
# args.desired_batch_size = [6, 6, 6, 6, 6, 6][exp_index]
# assert args.desired_batch_size % args.batch_size == 0

# args.update_interval = args.desired_batch_size // args.batch_size

# args.div_factor = 2
# args.learning_rate = 6e-05

# args.cuda_id = [0, 0, 0, 0, 0, 0][exp_index]

# args.num_steps = 60000 # Instead of number of epochs, define the number of desired steps.
# args.num_epochs = int(np.ceil(args.num_steps / (args.train_val_split[0] / args.desired_batch_size))) + 4
# # args.num_epochs = 12 # Use that when you want to specify the number of epochs, and not the number of steps.
# args.stop_after = args.num_epochs - 4 # Skip the very last epochs, where the learning rate it too small.
# args.rgb_only = [True, False, False, False, False, True][exp_index]
# args.input_channels = [3, 7, 7, 7, 7, 3][exp_index]
# args.early_stopping_thresh = 6

############################ Set default values, if not set otherwise ############################
 
args.hashtags_prefix = "####################################"
if args.desired_batch_size is None:
    args.desired_batch_size = args.batch_size
else:
    assert args.desired_batch_size % args.batch_size == 0, "Desired batch size must be a multiple of batch size"
    
args.update_interval = args.desired_batch_size // args.batch_size

if args.mini_dataset:
    assert args.run_mode == "test", "Mini dataset is only available for testing"
    
args.train_val_split = [0, 0] if args.mini_dataset else [17902, 2237]
    
args.num_samples = sum(args.train_val_split)
    
if args.num_steps is not None:
    args.num_epochs = args.num_steps * args.batch_size // args.train_val_split[0]
    print(f"{args.hashtags_prefix} Auto set number of epochs: {args.num_epochs}")
    
if args.checkpoint is not None:
    print(f"{args.hashtags_prefix} Auto set checkpoint to {args.checkpoint}")

args.split = Path("data") / args.split
assert args.split.exists(), f"Split file {args.split} does not exist"

if args.output_dir is not None:
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{args.hashtags_prefix} Auto set output directory to {args.output_dir}")
    
assert args.model in ["base", "base (rgb)", "supervised_seg", "unsupervised_seg", "sup_unsup_seg", "sup_unsup_seg (rgb)"], "Model type invalid"

args.supervised_seg = args.model in ["sup_unsup_seg", "sup_unsup_seg (rgb)", "supervised_seg"]
args.unsupervised_seg = args.model in ["sup_unsup_seg","sup_unsup_seg (rgb)", "unsupervised_seg"]
print(f"{args.hashtags_prefix} Auto set supervised_seg to {args.supervised_seg}")
print(f"{args.hashtags_prefix} Auto set unsupervised_seg to {args.unsupervised_seg}")
    

if args.model in ["base (rgb)", "sup_unsup_seg (rgb)"] or args.rgb_only:
    args.input_channels = 3
    print(f"{args.hashtags_prefix} Auto set input channels to 3 (RGB only)")
    assert args.input_channels == 3, "Input channels must be 3 for base_rgb model"
    
print(f"{args.hashtags_prefix} Auto set model to '{args.model}'")

if args.save_model:
    assert  args.run_name is not None, "If save_model is set, run_name must be set as well, to differentiate between different runs"

assert args.run_mode in ["train", "test"], "Run mode must be either train or test"

if args.checkpoint is not None and args.run_mode == "test":
    args.load_ckpt = True
    
if args.checkpoint is not None and not args.load_ckpt and args.run_mode == "train":
    user_input = input(f"{args.hashtags_prefix} Would you like to load the checkpoint file? [y/Y] for Yes, any other value for No. \n{args.hashtags_prefix} Answer: ")
    args.load_ckpt = user_input in ["y", "Y"]



# Model arguments
args.transformer_depths =  {"0": (2, 2, 2, 2), "1": (2, 2, 2, 2), "1.5": (2, 2, 3, 3), 
                           "2": (3, 3, 6, 3), "2.5": (3, 4, 7, 3), "3": (3, 6, 8, 3),
                           "3.5": (3, 8, 10, 3),  "4": (3, 8, 12, 5), "5": (3, 10, 16, 5)}["5"]

# Dataloader args
args.sparse_lidar = False
args.filtered_radar = False
args.lidar_ratio = [0.75, 0.25]
args.sparse_depth_uv = True
args.im_uv = False
args.rad_vel = True
args.radar_uv = False
args.gt_uv = False


