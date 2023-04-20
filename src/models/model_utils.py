import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torchvision.models.resnet import Bottleneck, conv1x1, conv3x3
import collections
import math
import params as params

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

class MultiTaskLossWrapper(nn.Module):
    def __init__(self):
        super(MultiTaskLossWrapper, self).__init__()

    def forward(self, preds, gt_depth, gt_seg):

        mse, crossEntropy = MaskedMSELoss(), nn.CrossEntropyLoss(ignore_index=255)

        loss_depth = mse(preds[0], gt_depth)
        loss_seg = crossEntropy(preds[1], gt_seg)
      
        return loss_depth, loss_seg    
    
class Weighted_MultiTaskLossWrapper(nn.Module):
    def __init__(self, model, eta):
        super(Weighted_MultiTaskLossWrapper, self).__init__()
        self.model= model
        self.eta = nn.Parameter(torch.Tensor(eta), requires_grad=True)

    def forward(self, input, gt_lidar_depth_stage1, gt_seg, gt_lidar_depth):

        preds = self.model(input)

        pred_depth_stage1 = preds[0][:,0,...].unsqueeze(1)
        pred_seg = preds[1]
        pred_depth_stage2 = preds[2]

        mse, crossEntropy = MaskedMSELoss(), nn.CrossEntropyLoss(ignore_index=255)

        loss_depth_stage1 = mse(pred_depth_stage1, gt_lidar_depth_stage1)
        loss_seg = 2* crossEntropy(pred_seg, gt_seg)
        loss_depth = mse(pred_depth_stage2, gt_lidar_depth)

        precision_depth_stage1 = torch.exp(-self.eta[0])
        loss_depth_stage1 = precision_depth_stage1*loss_depth_stage1 + self.eta[0]

        precision_seg = torch.exp(-self.eta[1])
        loss_seg = precision_seg*loss_seg + self.eta[1]

        precision_depth_stage2 = torch.exp(-self.eta[2])
        loss_depth = precision_depth_stage2*loss_depth + self.eta[2]

        loss = loss_depth_stage1 + loss_seg + loss_depth
        
        if params.error_map:
            pred_errormap = preds[0][:,1,...].unsqueeze(1)
            out_arr_depth_adapted = torch.where(gt_lidar_depth_stage1==0, gt_lidar_depth_stage1, pred_errormap) # Setting the predicted values to 0 where the Gt is invalid / has 0
            gt_errormap = out_arr_depth_adapted - gt_lidar_depth_stage1
            loss_error_map = mse(pred_errormap, gt_errormap)
            precision_error_map = torch.exp(-self.eta[3])
            loss_errormap = precision_error_map*loss_error_map + self.eta[3]
            loss += loss_errormap
            return loss, loss_depth_stage1, loss_seg, loss_depth, loss_errormap, preds
        else:
            return loss, loss_depth_stage1, loss_seg, loss_depth, preds

def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def weights_init_kaiming(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def weights_init_kaiming_leaky(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=None, dilation=1, norm_layer=None, res=True):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        padding = dilation
        self.conv1 = nn.Conv2d( in_channels=inplanes, out_channels=planes, kernel_size=kernel_size,
                                stride=stride, padding=padding, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True) #nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d( in_channels=planes, out_channels=planes, kernel_size=kernel_size,
                                stride=1, padding=padding, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.res = res

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        if self.res:
            out += identity
        out = self.relu(out)

        return out


class DSConv_Block(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=None, dilation=1, norm_layer=None, res=True):
        super(DSConv_Block, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        padding = dilation
        self.conv1 = Depthwise_Separable_Conv( in_planes=inplanes, out_planes=planes, kernel_size=kernel_size,
                                stride=stride, padding=padding, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True) #nn.ReLU(inplace=True)
        self.conv2 = Depthwise_Separable_Conv( in_planes=planes, out_planes=planes, kernel_size=kernel_size,
                                stride=1, padding=padding, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.res = res

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        if self.res:
            out += identity
        out = self.relu(out)

        return out

# Depthwise Separable Convolution --> much more computational efficient than normal conv layer
# from: https://github.com/seungjunlee96/Depthwise-Separable-Convolution_Pytorch
class Depthwise_Separable_Conv(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, dilation=1): 
        super(Depthwise_Separable_Conv, self).__init__() 
        self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_planes, dilation=dilation) 
        self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1, dilation=1) 
  
    def forward(self, x):

        out = self.depthwise(x) 
        out = self.pointwise(out) 

        return out

class Depthwise_Separable_Conv_Transposed(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=1, output_padding=1): 
        super(Depthwise_Separable_Conv_Transposed, self).__init__() 
        self.depthwise = nn.ConvTranspose2d(in_channels=in_planes, out_channels=in_planes, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=in_planes)
        self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1) 
    
    def forward(self, x):

        out = self.depthwise(x) 
        out = self.pointwise(out)
        
        return out


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



class Decoder(nn.Module):
    # Decoder is the base class for all decoders

    names = ['deconv2', 'deconv3', 'upconv', 'upproj', 'dsconv']

    def __init__(self):
        super(Decoder, self).__init__()

        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class DeConv(Decoder):
    def __init__(self, in_channels, kernel_size):
        assert kernel_size>=2, "kernel_size out of range: {}".format(kernel_size)
        super(DeConv, self).__init__()
        self.kernel_size = kernel_size
        self.layer1 = self.convt(in_channels, in_channels // 2)
        self.layer2 = self.convt(in_channels // 2, in_channels // 4)
        self.layer3 = self.convt(in_channels // 4, in_channels // 8)
        self.layer4 = self.convt(in_channels // 8, in_channels // 16)

    def convt(self, in_channels, out_channels):
        stride = 2
        padding = (self.kernel_size - 1) // 2
        output_padding = self.kernel_size % 2
        assert -2 - 2*padding + self.kernel_size + output_padding == 0, "deconv parameters incorrect"

        module_name = "deconv{}".format(self.kernel_size)
        return nn.Sequential(collections.OrderedDict([
                (module_name, nn.ConvTranspose2d(in_channels,out_channels, self.kernel_size,
                    stride,padding,output_padding,bias=False)),
                ('batchnorm', nn.BatchNorm2d(out_channels)),
                ('relu',      nn.ReLU(inplace=True)),
                ]))
    

class UpConv(Decoder):
    # UpConv decoder consists of 4 upconv modules with decreasing number of channels and increasing feature map size
    def upconv_module(self, in_channels, out_channels):
        # UpConv module: unpool -> 5*5 conv -> batchnorm -> ReLU
        upconv = nn.Sequential(collections.OrderedDict([
          ('unpool',    Unpool(in_channels)),
          ('conv',      nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False)),
          ('batchnorm', nn.BatchNorm2d(out_channels)),
          ('relu',      nn.ReLU()),
        ]))
        return upconv

    def __init__(self, in_channels):
        super(UpConv, self).__init__()
        self.layer1 = self.upconv_module(in_channels, in_channels//2)
        self.layer2 = self.upconv_module(in_channels//2, in_channels//4)
        self.layer3 = self.upconv_module(in_channels//4, in_channels//8)
        self.layer4 = self.upconv_module(in_channels//8, in_channels//16)

class UpProj(Decoder):
    # UpProj decoder consists of 4 upproj modules with decreasing number of channels and increasing feature map size

    class UpProjModule(nn.Module):
        # UpProj module has two branches, with a Unpool at the start and a ReLu at the end
        #   upper branch: 5*5 conv -> batchnorm -> ReLU -> 3*3 conv -> batchnorm
        #   bottom branch: 5*5 conv -> batchnorm

        def __init__(self, in_channels, out_channels):
            super(UpProj.UpProjModule, self).__init__()
            self.unpool = Unpool(in_channels)
            self.upper_branch = nn.Sequential(collections.OrderedDict([
              ('conv1',      nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False)),
              ('batchnorm1', nn.BatchNorm2d(out_channels)),
              ('relu',       nn.LeakyReLU(0.2)), #nn.ReLU()),
              ('conv2',      nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)),
              ('batchnorm2', nn.BatchNorm2d(out_channels)),
            ]))
            self.bottom_branch = nn.Sequential(collections.OrderedDict([
              ('conv',      nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False)),
              ('batchnorm', nn.BatchNorm2d(out_channels)),
            ]))
            self.relu = nn.LeakyReLU(0.2) #nn.ReLU()

        def forward(self, x):
            x = self.unpool(x)
            x1 = self.upper_branch(x)
            x2 = self.bottom_branch(x)
            x = x1 + x2
            x = self.relu(x)
            return x

    def __init__(self, in_channels):
        super(UpProj, self).__init__()
        self.layer1 = self.UpProjModule(in_channels, in_channels//2)
        self.layer2 = self.UpProjModule(in_channels//2, in_channels//4)
        self.layer3 = self.UpProjModule(in_channels//4, in_channels//8)
        self.layer4 = self.UpProjModule(in_channels//8, in_channels//16)

class DSConvT(Decoder):
    def __init__(self, in_channels, kernel_size):
        assert kernel_size>=2, "kernel_size out of range: {}".format(kernel_size)
        super(DSConvT, self).__init__()
        self.kernel_size = kernel_size
        self.layer1 = self.dsconvt(in_channels, in_channels // 2)
        self.layer2 = self.dsconvt(in_channels // 2, in_channels // 4)
        self.layer3 = self.dsconvt(in_channels // 4, in_channels // 8)
        self.layer4 = self.dsconvt(in_channels // 8, in_channels // 16)

    def dsconvt(self, in_channels, out_channels):
        stride = 2
        padding = (self.kernel_size - 1) // 2
        output_padding = self.kernel_size % 2
        assert -2 - 2*padding + self.kernel_size + output_padding == 0, "deconv parameters incorrect"

        module_name = "deconv{}".format(self.kernel_size)
        return nn.Sequential(collections.OrderedDict([
                (module_name, Depthwise_Separable_Conv_Transposed(in_channels,out_channels, self.kernel_size,
                    stride,padding,output_padding)),
                ('batchnorm', nn.BatchNorm2d(out_channels)),
                ('relu',      nn.ReLU(inplace=True)),
                ]))


def choose_decoder(decoder, in_channels):
    # iheight, iwidth = 10, 8
    if decoder[:6] == 'deconv':
        assert len(decoder)==7
        kernel_size = int(decoder[6])
        return DeConv(in_channels, kernel_size)
    elif decoder == "upproj":
        return UpProj(in_channels)
    elif decoder == "upconv":
        return UpConv(in_channels)
    elif decoder == "dsconv":
        return DSConvT(in_channels, kernel_size=2)
    else:
        assert False, "invalid option for decoder: {}".format(decoder)