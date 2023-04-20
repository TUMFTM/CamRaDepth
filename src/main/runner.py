from copy import deepcopy
import os, sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), '..',))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..')) 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import datetime
import shutil
from pathlib import Path
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import numpy as np
from tqdm import tqdm
from utils.args import args 
from data.dataloader import make_dataloaders
from torchmetrics import JaccardIndex

from torch.utils.tensorboard import SummaryWriter
from models.diffGradNorm import diffGradNorm
from models.CamRaDepth import CamRaDepth
# from src.train_with_tranformers.mlt_fscrach_seg import CamRaDepth



from utils.utils import load_checkpoint_with_shape_match, create_tqdm_bar
from utils.loss_funcs import MaskedFocalLoss, MaskedSmoothL1Loss, MaskedMSELoss

torch.backends.cudnn.benchmark = True


def save_files(model, output_path):
    
    """
    If you decide to use this functionality, you'll have to set the relevant paths first.
    """
    
    project_files_path = Path(output_path) / "project_files"
    os.makedirs(project_files_path, exist_ok=True)
    
    this_dir = os.path.dirname(__file__)
    
    # Model:
    model_file = None
    assert model, "Model is None"
    if type(model) == CamRaDepth:
        model_file = os.path.join(this_dir, "../models/CamRaDepth.py")
    else:
        raise ValueError("Model type not supported")
    
    model_file_dst = project_files_path / Path(model_file).name
    shutil.copyfile(model_file, model_file_dst)
    
    # Transformer backbone:
    transformer_backbone_file = os.path.join(this_dir, "../models/simplified_attention.py")
    transformer_backbone_file_dst = project_files_path / Path(transformer_backbone_file).name
    shutil.copyfile(transformer_backbone_file, transformer_backbone_file_dst)
    
    # Args:
    args_path = os.path.join(this_dir, "../utils/args.py")
    args_dst = project_files_path / Path(args_path).name
    shutil.copyfile(args_path, args_dst)
    
    # Dataloader:
    dataloader_path = os.path.join(this_dir, "../data/dataloader.py")
    dataloader_path_dst = project_files_path / Path(dataloader_path).name
    shutil.copyfile(dataloader_path, dataloader_path_dst)
    
    # Utils:
    utils_path = os.path.join(this_dir, "../utils/utils.py")
    utils_path_dst = project_files_path / Path(utils_path).name
    shutil.copyfile(utils_path, utils_path_dst)
    
    # Splitfile:
    splitfile_path = os.path.join(this_dir, "../data/original_split.npy")
    splitfile_path_dst = project_files_path / Path(splitfile_path).name
    shutil.copyfile(splitfile_path, splitfile_path_dst)
    
    # Runner File:
    train_path = splitfile_path = os.path.join(this_dir, "../main/runner.py")
    train_path_dst = project_files_path / Path(train_path).name
    shutil.copyfile(train_path, train_path_dst)
    

class Trainer:
    
    def __init__(self, model, save=True, mode="train") -> None:
        
        self.device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')
        self.training_steps = 0
        self.val_steps = 0
    
        print(f"{args.hashtags_prefix} Using device: {self.device}")
        self.model = model(input_channels=args.input_channels).to(self.device)
        
        if save:
            out = Path(args.output_dir) / Path(args.arch_name)
            os.makedirs(out, exist_ok=True)
            dirs = os.listdir(out)
            dirs += ["0"]
            index = str(max([int(x) for x in dirs if x.isdigit()]) + 1)

            path = out / index if args.run_name is None else out / args.run_name
            os.makedirs(path, exist_ok=True)
            
            dirs = os.listdir(path)
            dirs += ["0"]
            index = str(max([int(x) for x in dirs if x.isdigit()]) + 1)
            path = path / index 
            os.makedirs(path, exist_ok=True)
            
            self.new_log_path = path 
            self.new_run_path =  path
            self.tb_logger = SummaryWriter(path, flush_secs=10)
            
            save_files(self.model, self.new_run_path)

        
        if mode == "test" and (args.checkpoint is None):
            raise ValueError("A checkpoint is needed for testing!")
        
        if args.checkpoint is not None and args.load_ckpt:
            if os.path.exists(args.checkpoint):
                device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')
                state = torch.load(args.checkpoint, map_location=device)
                load_checkpoint_with_shape_match(self.model, state["state_dict"])
                args.learning_rate = state.get("lr", args.learning_rate)
                print(f"{args.hashtags_prefix} Loaded checkpoint from {args.checkpoint}")
            else:
                raise ValueError(f"Checkpoint not found at {os.path.abspath(args.checkpoint)}!")   
        
        self.model = self.model.to(self.device)
        if args.distributed:
            self.model = nn.DataParallel(self.model)
            
        ######################################## Debugging purpuses
        # dataloaders = make_dataloaders(num_samples=100, train_part=0.8, split=mode)
        dataloaders = make_dataloaders(mode)
        if mode == "train":
            print(f"{args.hashtags_prefix} Mode: Train")
            if args.num_steps is not None:
                print(f"{args.hashtags_prefix} No. steps: {args.num_steps}")
            print(f"{args.hashtags_prefix} No. epochs: {args.num_epochs}")
    
            self.train_dataloader, self.val_dataloader = dataloaders["train"], dataloaders["val"]
            
            self.criterion = {'depth': MaskedSmoothL1Loss(), 'seg': MaskedFocalLoss()}
            self.optimizer = diffGradNorm(self.model.parameters(), lr=args.learning_rate)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.learning_rate, steps_per_epoch=len(self.train_dataloader), 
                                                             epochs=args.num_epochs, div_factor=args.div_factor, pct_start=0.15) # Change back to 0.05
        else:
            print(f"{args.hashtags_prefix} Mode: Test")
            self.test_dataloader = dataloaders["test"]
            args.arch_name = "Inference"
            args.run_name = "Inference"
                    
        self.scaler = torch.cuda.amp.GradScaler()
       
        
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
    
    def train_one_epoch(self, epoch, save=True):
        self.model.train()
        loop = create_tqdm_bar(self.train_dataloader, f"Training [{epoch + 1}/{args.num_epochs}]")
        
        all_losses = []
        rmse_arr = []
        depth_final_loss_arr = []
        depth_stage_1_loss_arr = []
        seg_loss_arr = []
        self.optimizer.zero_grad() 
        
        cur_lr = self.optimizer.param_groups[0]['lr']
        
        for i, batch in loop:
            feature_tensor = batch["image"].to(torch.float32).cuda()
            inputs = Variable(feature_tensor).cuda()
            gt = batch["gt"]
            
            gt_depth_final = gt["depth"]['lidar_depth'].to(torch.float32).cuda()
            gt_depth_stage_4, gt_depth_stage_3, _ = gt["depth"]['lidar_depth_partial']
            gt_depth_stage_4, gt_depth_stage_3 = gt_depth_stage_4.to(torch.float32).cuda(), \
                gt_depth_stage_3.to(torch.float32).cuda()
                
            gt_seg_final = gt['seg']["final_seg"].to(torch.long).cuda()
            gt_seg_inter = gt['seg']["intermediate_seg"].to(torch.long).cuda()
            with torch.autocast('cuda'):
                
                pred_dict = self.model(inputs[:, :args.input_channels])
                depth_stage_full, inter_depths = pred_dict["depth"]['final_depth'], pred_dict["depth"]['intermediate_depths']
                final_seg, inter_seg = pred_dict['seg']['final_seg'], pred_dict['seg']['intermediate_seg']
                
                loss_seg_final = (self.criterion['seg'](final_seg, gt_seg_final) if final_seg is not None else 0) * args.supervised_seg
                loss_seg_inter = (self.criterion['seg'](inter_seg, gt_seg_inter) if inter_seg is not None else 0) * args.supervised_seg
                seg_loss_arr.append(loss_seg_final.item() if type(loss_seg_final) == torch.Tensor else loss_seg_final)
                
                loss_depth_stage_4 = self.criterion['depth'](inter_depths[-1].squeeze(1), gt_depth_stage_4.squeeze(1))
                loss_depth_stage_3 = self.criterion['depth'](inter_depths[-2].squeeze(1), gt_depth_stage_3.squeeze(1))
                loss_depth_final = self.criterion['depth'](depth_stage_full, gt_depth_final)
            
            depth_stage_1_loss_arr.append(loss_depth_stage_4.item())
            depth_final_loss_arr.append(loss_depth_final.item())
            
            RMSE = torch.sqrt(MaskedMSELoss()(depth_stage_full, gt_depth_final)).item() * args.max_depth
            rmse_arr.append(RMSE)
           
            ### Backward pass and optimizer
            
            losses_weights = [1, 1, 1, 0.2 , 0.2]
            loss = (losses_weights[0] * loss_depth_final  + losses_weights[1] * loss_depth_stage_4 + losses_weights[2] * loss_depth_stage_3 + 
                     + losses_weights[3] * loss_seg_final + losses_weights[4] * loss_seg_inter) / sum(losses_weights)
            
            # calculates gradients
            loss = loss / args.update_interval # Accumulated gradients need to be divided by the number of iterations
            self.scaler.scale(loss).backward()
            
            # Perform the optimizer step after the required number of iterations for accumelated gradients (Could also be simply 1).
            progess_bool = (i + 1) % args.update_interval == 0 or (i + 1) == len(self.train_dataloader)
            
            if progess_bool:
                
                # Update the progress tqdm bar and Tensorboard.
                cutoff = 600
                cur_lr = self.optimizer.param_groups[0]['lr']
                loss_depth_final = np.nanmean(depth_final_loss_arr)
                loss_depth_stage_4 = np.nanmean(depth_stage_1_loss_arr)
                rmse_mean = np.nanmean(rmse_arr)
                depth_final_loss_arr, depth_stage_1_loss_arr, rmse_arr = [], [], []
                
                loss_seg = np.nanmean(seg_loss_arr)
                seg_loss_arr = []
                
                all_losses.append([loss_depth_final, rmse_mean, loss_seg])
                all_losses = all_losses[-cutoff:]
                
                mean_losses = np.nanmean(all_losses, axis=0)
                
                depth_mean = mean_losses[0]
                rmse_mean = mean_losses[1]
                seg_mean = mean_losses[2]
                
                loop.set_postfix(
                                    learning_rate = "{:.7f}".format(cur_lr),
                                    depth_mean =  "{:.6f}".format(depth_mean),
                                    depth_loss =  "{:.6f}".format(loss_depth_final),
                                    RMSE = "{:.6f}".format(rmse_mean),
                                    seg_mean = "{:.6f}".format(seg_mean)
                                )
                if save:
                    try:
                        self.tb_logger.add_scalar(f"{args.arch_name}/train_mean_seg", seg_mean, self.training_steps)
                        self.tb_logger.add_scalar(f"{args.arch_name}/train_loss_depth", loss_depth_final, self.training_steps)
                        self.tb_logger.add_scalar(f"{args.arch_name}/train_mean_depth", depth_mean , self.training_steps)
                        self.tb_logger.add_scalar(f"{args.arch_name}/learning Rate", cur_lr, self.training_steps)
                        self.tb_logger.add_scalar(f"{args.arch_name}/RMSE", rmse_mean, self.training_steps)
                    except OSError:
                        print("OSError, tensorboard not available")
                    
                self.training_steps += 1
                self.scaler.step(self.optimizer) 
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            
            # To prevent a scheduler step before the optimizer step.
            if (i + 1) > args.update_interval:
                self.scheduler.step()
                    

    def eval(self, epoch, save=True):
        self.model.eval()
        eval_losses = []
        rmse_arr = []
        depth_final_loss_arr = []
        depth_stage_1_loss_arr = []
        seg_loss_arr = []
        
        with torch.no_grad():
            loop_eval = create_tqdm_bar(self.val_dataloader, f"Val [{epoch + 1}/{args.num_epochs}]") # tqdm is needed for visualization of the training progess in the terminal
            for i, batch in loop_eval:
                feature_tensor = batch["image"].to(torch.float32).cuda()
                inputs = Variable(feature_tensor).cuda()
                gt = batch["gt"]
                inputs = Variable(feature_tensor).cuda()
                
                gt_depth_final = gt["depth"]['lidar_depth'].to(torch.float32).cuda()
                gt_depth_stage_4, gt_depth_stage_3, gt_depth_stage_2 = gt["depth"]['lidar_depth_partial']
                gt_depth_stage_4, gt_depth_stage_3, gt_depth_stage_2 = gt_depth_stage_4.to(torch.float32).cuda(), \
                    gt_depth_stage_3.to(torch.float32).cuda(), gt_depth_stage_2.to(torch.float32).cuda()
                    
                gt_seg_final = gt['seg']["final_seg"].to(torch.long).cuda()

                with torch.autocast("cuda"):
                    pred_dict = self.model(inputs[:, :args.num_features])
                    depth_stage_full, inter_depths = pred_dict["depth"]['final_depth'], pred_dict["depth"]['intermediate_depths']
                    final_seg, _ = pred_dict['seg']['final_seg'], pred_dict['seg']['intermediate_seg']
                    
                    loss_seg_final = self.criterion['seg'](final_seg, gt_seg_final) if final_seg is not None else 0
                    seg_loss_arr.append(loss_seg_final.item() if type(loss_seg_final) == torch.Tensor else loss_seg_final)
                    
                    loss_depth_stage_4 = self.criterion['depth'](inter_depths[-1].squeeze(1), gt_depth_stage_4.squeeze(1)).item()
                    loss_depth_final = self.criterion['depth'](depth_stage_full, gt_depth_final).item()
                
                RMSE = torch.sqrt(MaskedMSELoss()(depth_stage_full, gt_depth_final)).item() * args.max_depth
                rmse_arr.append(RMSE) 
                depth_final_loss_arr.append(loss_depth_final)
                depth_stage_1_loss_arr.append(loss_depth_stage_4)
                
                progess_bool = (i + 1) % args.update_interval == 0 or (i + 1) == len(self.val_dataloader)
                if progess_bool:
                    cutoff = 600
                    
                    val_rmse = np.nanmean(rmse_arr[-cutoff:])
                    depth_final_loss = np.nanmean(depth_final_loss_arr)
                    depth_inter_loss = np.nanmean(depth_stage_1_loss_arr)
                    seg_loss = np.nanmean(seg_loss_arr)
                    depth_final_loss_arr, depth_stage_1_loss_arr, seg_loss_arr = [], [], []
                    
                    eval_losses.append([depth_final_loss, depth_inter_loss, val_rmse, seg_loss])
                    cut_eval_losses = eval_losses[-cutoff:]
                    losses_mean = np.nanmean(cut_eval_losses[-cutoff:], axis=0)
                    depth_mean = losses_mean[0]
                    inter_depth_mean = losses_mean[1]
                    rmse_mean = losses_mean[2]
                    seg_mean = losses_mean[3]                   
                    
                    loop_eval.set_postfix(
                        v_depth_mean = depth_mean,
                        v_inter_depth_mean = inter_depth_mean,
                        v_seg_mean = seg_mean,       
                        v_RMSE_mean = rmse_mean
                    )
                    if save:
                        try:
                            self.tb_logger.add_scalar(f"{args.arch_name}/val_depth", depth_final_loss, self.val_steps)
                            self.tb_logger.add_scalar(f"{args.arch_name}/val_mean_depth", depth_mean, self.val_steps)
                            self.tb_logger.add_scalar(f"{args.arch_name}/val_mean_inter_depth", inter_depth_mean, self.val_steps)
                            self.tb_logger.add_scalar(f"{args.arch_name}/val_RMSE", rmse_mean, self.val_steps)
                            self.tb_logger.add_scalar(f"{args.arch_name}/val_seg_seg", seg_mean, self.val_steps)
                        except OSError:
                            print("OSError, tensorboard not available")
                    self.val_steps += 1

        eval_losses = np.array(eval_losses)
        val_loss = np.nanmean(eval_losses[:, 0])
        RMSE = np.nanmean(eval_losses[:, 2])       
        return val_loss, RMSE

    def train(self, save=True):

        start_training = time.time()
        early_stop_counter = 0
        best_eval_loss = np.inf

        # Create the models saving folder for the current run
        
        for epoch in range(args.num_epochs):
            self.train_one_epoch(epoch=epoch, save=save)
            eval_loss, RMSE = self.eval(epoch=epoch, save=save)
            print("{args.hashtags_prefix} Eval loss: ", eval_loss, "RMSE: ", RMSE)

            # Early stopping
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                if save:
                    state = {'state_dict': self.model.to('cpu').state_dict(), 'optimizer': self.optimizer.state_dict(), "lr": self.optimizer.param_groups[0]['lr'], "steps": [self.training_steps, self.val_steps],}
                    path = os.path.join(self.new_run_path, "mlt" + '_epoch_' + str(epoch+1) +"_best_eval_loss_" + "{:.7f}".format(best_eval_loss.item()) +'.pth')
                    torch.save(state, path)
                    self.model.to(device=self.device)
                    print('{args.hashtags_prefix} Model saved to {}'.format(self.new_run_path))
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            if early_stop_counter == args.early_stopping_thresh:
                print("{args.hashtags_prefix} Early stopping to prevent overfitting")
                break
            
            if epoch + 1 == args.get("stop_after", args.num_epochs - 4):
                print("{args.hashtags_prefix} Training finished")
                exit()
                
                
        stop_training = time.time()

        print('Training done.')
        print('Time for total training    : ', str(datetime.timedelta(seconds=stop_training - start_training)))

        return best_eval_loss

    def test(self, save=True):
        this_dir = os.path.dirname(__file__)
        output_path = Path(os.path.join(this_dir, "src/Output")) / "inference"
        if os.path.exists(output_path):
            shutil.rmtree(output_path, ignore_errors=True)
        os.makedirs(output_path, exist_ok=True)
        
        self.model.eval()
        with torch.no_grad():
            loop_test = create_tqdm_bar(self.test_dataloader, "Test")
            times = []
            metrics_100 = []
            metrics_50 = []
            edge = []
            sunny = []
            for k, batch in loop_test:
                
                ### Extract batch
                feature_tensor = batch["image"].to(torch.float32).cuda()
                inputs = Variable(feature_tensor).cuda()
            
                ### Forward pass
                 
                start = time.time()
                pred_dict = self.model(inputs[:, :args.input_channels])
                end = time.time()
                inference_time = end-start
                
                ### Extract ground truth
                gt = batch["gt"]
                gt_seg = gt['seg']["final_seg"].to(torch.long)
                gt_depth = gt["depth"]['lidar_depth'].to(torch.float32).cuda() 
                
                
                depth_stage_full, _ = pred_dict["depth"]['final_depth'], pred_dict["depth"]['intermediate_depths']
                pred_seg, _ = pred_dict['seg']['final_seg'], pred_dict['seg']['intermediate_seg']
                
                ### Segmentation
                IoU = np.nan
                if args.supervised_seg and pred_seg is not None:
                    try:
                        jaccard = JaccardIndex(num_classes=args.num_classes, ignore_index=255).to(device=self.device)
                        IoU = jaccard(pred_seg.to(device=self.device), gt_seg.to(device=self.device)).detach().cpu().numpy()
                        pred_seg = torch.argmax(pred_seg, dim=1)
                    except ValueError as e:
                        pass
                                                
                        
                ### Final depth (416x800) 
                pred_depth = depth_stage_full.squeeze()
                pred_depth = torch.clip(pred_depth, 0, 1)
                pred = pred_depth
                gt = gt_depth.squeeze()
                
                pred = args.max_depth * pred
                gt = args.max_depth * gt
        
                # Only consider depths within max range, but remeber that this is inverse.
                indices = gt > args.max_distances[0] 
                gt[indices] = 0
                indices = torch.where(gt > 0)
                if len(indices[0]) == 0:
                    continue
                
                errormap = pred[indices] - gt[indices]
                rel_errormap = torch.abs(errormap) / gt[indices]
                
                l2 = nn.MSELoss()
                l1 = nn.L1Loss()
                MAE = l1(pred[indices], gt[indices]).item() 
                RMSE = torch.sqrt(l2(pred[indices], gt[indices])).item()
                REL = (torch.sum(rel_errormap) / len(rel_errormap)).item() 
                
                pred_depth = pred_depth.cpu().numpy()
                if "rain" in batch["name"][0]:
                    edge.append(RMSE)
                else:
                    sunny.append(RMSE)
                    
                times.append(inference_time)
                metrics_100.append([RMSE, MAE, REL, IoU])
                loop_test.set_postfix(RMSE = RMSE, MAE = MAE, REL = REL, RMSE_mean = np.nanmean(np.array(metrics_100)[:, 0]), edge = np.nanmean(np.array(edge)), sunny = np.nanmean(np.array(sunny)))
                
                indices =  gt < args.max_distances[1]
                gt[indices] = 0
                indices = torch.where(gt > 0)
                if len(indices[0]) == 0:
                    continue
                
                errormap = pred[indices] - gt[indices]
                rel_errormap = torch.abs(errormap) / gt[indices]
                
                l2 = nn.MSELoss()
                l1 = nn.L1Loss()
                MAE = l1(pred[indices], gt[indices]).item() 
                RMSE = torch.sqrt(l2(pred[indices], gt[indices])).item() 
                REL = (torch.sum(rel_errormap) / len(rel_errormap)).item() 
            
                metrics_50.append([RMSE, MAE, REL])
                
        print(f"{args.hashtags_prefix} max depth {args.max_distances[0]} {args.hashtags_prefix}")
        time_mean = np.nanmean(np.array(times))
        rmse_mean = np.nanmean(np.array(metrics_100)[:, 0])
        mae_mean = np.nanmean(np.array(metrics_100)[:, 1])
        rel_mean = np.nanmean(np.array(metrics_100)[:, 2])
        edge_mean = np.nanmean(np.array(edge))
        sunny_mean = np.nanmean(np.array(sunny))
        print('The inference time is:   ', time_mean, 's')
        print('The RMSE of the predicted depth is:   ', rmse_mean, 'meter(s)')
        print("The sunny error is: ", sunny_mean, "meter(s)")
        print("The edge error is: ", edge_mean, "meter(s)")
        print('The MAE of the predicted depth is:    ', mae_mean, 'meter(s).')
        print('The REL of the predicted depth is:    ', rel_mean)
        if args.supervised_seg:
            iou_mean = np.nanmean(np.array(metrics_100)[:, 3])
            print('The IoU of the predicted segmentation is:    ', iou_mean)
        
        print(f"{args.hashtags_prefix} max depth {args.max_distances[1]} {args.hashtags_prefix}")
        time_mean = np.nanmean(np.array(times))
        rmse_mean = np.nanmean(np.array(metrics_50)[:, 0])
        mae_mean = np.nanmean(np.array(metrics_50)[:, 1])
        rel_mean = np.nanmean(np.array(metrics_50)[:, 2])
        print('The inference time is:   ', time_mean, 's')
        print('The RMSE of the predicted depth is:   ', rmse_mean, 'meter(s)')
        print('The MAE of the predicted depth is:    ', mae_mean, 'meter(s).')
        print('The REL of the predicted depth is:    ', rel_mean)
            
    def hyperparameters_tuning(self, **kwargs):

        def random_search_spaces_to_config(random_search_spaces):
            """"
            Takes search spaces for random search as input; samples accordingly
            from these spaces and returns the sampled hyper-args as a config-object,
            which will be used to construct solver & network
            """
            config = {}
            for key, (rng, mode)  in random_search_spaces.items():
                if mode not in ["log", "int", "float", "item", "fixed"]:
                    print("'{}' is not a valid random sampling mode. "
                        "Ignoring hyper-param '{}'".format(mode, key))
                elif mode == "log":
                    if rng[0] <= 0 or rng[-1] <=0:
                        print("Invalid value encountered for logarithmic sampling "
                            "of '{}'. Ignoring this hyper param.".format(key))
                        continue
                    sample = np.random.uniform(np.log10(rng[0]), np.log10(rng[-1]))
                    config[key] = 10**(sample)
                elif mode == "int":
                    config[key] = np.random.randint(rng[0], rng[-1])
                elif mode == "float":
                    config[key] = np.random.uniform(rng[0], rng[-1])
                elif mode == "item":
                    config[key] = rng[int(np.random.choice(np.arange(len(rng))))]
            return config

        configs = []
        for i in range(args.random_search_num_trials):
            configs.append(random_search_spaces_to_config(kwargs))
        
        best_eval_loss = np.inf
        best_config = None
        for i in range(len(configs)):

            pass # TODO: write relevant code for hyperparameters tuning

            if curr_eval_loss < best_eval_loss:
                print("Found a better config!")
                best_eval_loss = curr_eval_loss
                best_config = config

            del self.model
        print("Best config: ", best_config)
        
        

if __name__ == "__main__":
    
    
    # For easily tweaking the arguments. One could also follow the more conventional way of passing arguments to the command line,
    # like a medieval peasent.
    
    with torch.cuda.device(args.cuda_id):
        if args.run_mode == "train":
            Trainer(model=CamRaDepth, save=args.save_model).train()
        elif args.run_mode == "test":
            Trainer(model=CamRaDepth, save=args.save_model, mode=args.run_mode).test()
        else:
            raise ValueError("Invalid run mode. Please choose between 'train' and 'test'.")

        