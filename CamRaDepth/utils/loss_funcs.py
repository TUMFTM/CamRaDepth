import sys, os
sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.path.abspath(os.path.dirname(__file__)) + '/' + '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.args import args
import cv2
import numpy as np



class MaskedFocalLoss(nn.modules.loss._WeightedLoss):
    ''' Focal loss for classification tasks on imbalanced datasets '''

    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super().__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(ignore_index=255)
        self.eps = 1e-8

    def forward(self, inputs, target):
        
        ce_loss = self.ce(inputs, target)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
    
class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff**2).mean()
        return self.loss


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss

class MaskedHuberLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.crieterion = torch.nn.HuberLoss()


    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()

        # Mask out the content
        pred = pred[valid_mask]
        target = target[valid_mask]

        return self.crieterion(pred, target)

class MaskedSmoothL1Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.SmoothL1Loss()
        self.eps = 1e-8

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()

        # Mask out the content
        pred = pred[valid_mask]
        target = target[valid_mask]

        return self.criterion(pred, target)
    
    
class MaskedSSIMLoss(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.criterion = SSIM(data_range=1.0, size_average=True, channel=1)
    
    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        
        old_shape = target.shape
        target = target.cpu().detach().numpy().squeeze()
        print(target.shape)
        
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        target = cv2.filter2D(target, -1, kernel).reshape(old_shape)
        target = torch.from_numpy(target).cuda().type(pred.type())
        
        # Mask out the content
        pred = pred[valid_mask]
        target = target[valid_mask]

        return 1 - self.criterion(pred, target)

class MaskedRMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = torch.sqrt((diff**2).mean())
        return self.loss

class MaskedBerHuLoss(nn.Module):
    def __init__(self, thresh=0.2):
        super(MaskedBerHuLoss, self).__init__()
        self.thresh = thresh

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()

        # Mask out the content
        pred = pred[valid_mask]
        target = target[valid_mask]

        # ipdb.set_trace()
        diff = torch.abs(target - pred)
        delta = self.thresh * torch.max(diff).item()

        part1 = - torch.nn.functional.threshold(-diff, -delta, 0.)
        part2 = torch.nn.functional.threshold(diff ** 2 - delta ** 2, 0., -delta**2.) + delta ** 2
        part2 = part2 / (2. * delta)

        loss = part1 + part2
        loss = torch.mean(loss)

        return loss

# Define the smoothness loss
class SmoothnessLoss(nn.Module):
    def __init__(self):
        super(SmoothnessLoss, self).__init__()
    
    def forward(self, pred_depth, image):

        image = image.to(torch.float32)
        # Normalize the depth with mean
        depth_mean = pred_depth.mean(2, True).mean(3, True)
        pred_depth_normalized = pred_depth / (depth_mean + 1e-7)

        # Compute the gradient of depth
        grad_depth_x = torch.abs(pred_depth_normalized[:, :, :, :-1] - pred_depth_normalized[:, :, :, 1:])
        grad_depth_y = torch.abs(pred_depth_normalized[:, :, :-1, :] - pred_depth_normalized[:, :, 1:, :])

        # Compute the gradient of the image
        grad_image_x = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), 1, keepdim=True)
        grad_image_y = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), 1, keepdim=True)

        grad_depth_x *= torch.exp(-grad_image_x)
        grad_depth_y *= torch.exp(-grad_image_y)

        return grad_depth_x.mean() + grad_depth_y.mean()


class Unpool(nn.Module):
    # Unpool: 2*2 unpooling with zero padding
    def __init__(self, num_channels, stride=2):
        super(Unpool, self).__init__()

        self.num_channels = num_channels
        self.stride = stride

        # create kernel [1, 0; 0, 0]

        self.weights = torch.autograd.Variable(torch.zeros(num_channels, 1, stride, stride).cuda()) # currently not compatible with running on CPU
        self.weights[:,:,0,0] = 1

    def forward(self, x):
        return F.conv_transpose2d(x, self.weights, stride=self.stride, groups=self.num_channels)

    
   