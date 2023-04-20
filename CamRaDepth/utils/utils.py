import sys, os

from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..', ".."))
from utils.args import args
import torch.nn as nn
import torch
import math


 
class AttentionBlcok(nn.Module):
    """
    Creates an attention vector at a desired length, corresponding to an outer-scope feature maps block.
    The learned attention is between the latter different channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        out = self.average_pooling(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = torch.sigmoid(out)
        return out
    
class SparaseDenseLayer(nn.Module):
    """
    Has two branches: A convolution, and an attention vector that learns the correspondences between the convolution's output
    channels.
    """
    def __init__(self, in_channels, out_channels, mid_channels=128, maxpool_bool=False, dense=False, as_final_block=False):
        super().__init__()
        self.as_final_block = as_final_block
        self.conv3x3 = ConvLayer(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.atten = AttentionBlcok(in_channels, mid_channels)
        
        if as_final_block:
            self.conv_combine = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.conv_combine = ConvLayer(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
      
            
    def forward(self, x):
        out = self.conv3x3(x)
        atten = self.atten(x)
        out = out * atten + out
        out = self.conv_combine(out)
        return out
    
    
class SparaseDenseBlock(nn.Module):
    
    """
    A sub-network of attention blocks.
    """
    
    def __init__(self, in_channels, out_channels, mid_channels=128, num_layers=1, as_final_block=False, **kwargs):
        super().__init__()
        self.as_final_block = as_final_block
        self.num_layers = num_layers
        # self.stem = ConvLayer(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.layers = nn.ModuleList()
        
        half_channels = mid_channels
        inp = in_channels
        for i in range(num_layers):
            out = half_channels if i < (num_layers - 1) else out_channels
            as_final = as_final_block and i == (num_layers - 1)
            self.layers.append(
                 SparaseDenseLayer(inp, out, mid_channels=mid_channels, as_final_block=as_final)
            )
            inp += out
        
    def forward(self, x):
        # x = self.stem(x)
        for layer in self.layers[:-1]:
            out = layer(x)
            x = torch.cat((x, out), dim=1)
        x = self.layers[-1](x)
        return x
            

class Seg_Block(nn.Module):
    """
    Creates a segmnetation map out of an input block of logits.
    """
    def __init__(self, num_classes=21):
        super().__init__()
        self.seg_num_classes = num_classes
    
    def forward(self, seg_logits):
        seg_map =  torch.argmax(seg_logits, dim=1, keepdim=True)
        seg_map = seg_map / self.seg_num_classes
        
    
        return seg_map


class ShortResBlock(nn.Module):
    """
    A short dense blocks, with reducing channels as it goes deeper.
    """
    def __init__(self, in_channels, out_channels, mid_channels=128, num_layers=3, maxpool_bool=False, as_final_block=False, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.maxpool_bool = maxpool_bool
        
        self.layers = nn.ModuleList()
        
        multi_factor = 0.75
        inp = in_channels
        out = int(mid_channels * multi_factor)
        for i in range(num_layers): 

            self.layers.append(
                ConvLayer(inp, out, kernel_size=3, stride=1, padding=1)
            )
            inp += out
            multi_factor -= 0.25
            out = out_channels if i == num_layers - 2 else int(mid_channels * multi_factor)
            
    
    def forward(self, x):
        # print(x.shape)
        for layer in self.layers[:-1]:
            out = layer(x)
            # print(out.shape)
            x = torch.cat((x, out), dim=1)
            # print(x.shape)
        x = self.layers[-1](x)
        return x

class ResBlock(nn.Module):
    """
    A full on dense block - might be too expansive to use. The short version is a much preferred option.
    Can be implemented as dense block, or as a regular resblock.
    """
    def __init__(self, in_channels, out_channels, mid_channels=128, maxpool_bool=False, dense=False, as_final_block=False, **kwargs):
        super().__init__()
        
        self.dense = dense
        self.maxpool_bool = maxpool_bool
                
        if dense:
            self.block_1 = ConvLayer(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
            self.block_2 = ConvLayer(in_channels + mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
            self.block_3 = ConvLayer(in_channels + mid_channels + mid_channels, mid_channels, kernel_size=1, stride=1, padding=0)
            
            if as_final_block:
                self.block_4 = nn.Sequential(
                    ConvLayer(in_channels + mid_channels + mid_channels + mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
                )
            else:
                self.block_4 = ConvLayer(in_channels + mid_channels + mid_channels + mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.block_1 = ConvLayer(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
            self.block_2 = ConvLayer(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
            self.block_3 = ConvLayer(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0)
            
            if as_final_block:
                self.block_4 = nn.Sequential(
                    ConvLayer(mid_channels + in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
                )
            else:
                self.block_4 = ConvLayer(mid_channels + in_channels, out_channels, kernel_size=3, stride=1, padding=1)
                
        if  self.maxpool_bool:
            self.maxpool = nn.MaxPool2d(2)
            
        self.addition = nn.quantized.FloatFunctional()


    def forward(self, x):
        if self.dense:
            out = self.block_1(x)
            x = torch.cat((x, out), 1) 
            out = self.block_2(x)
            x = torch.cat((x, out), 1) 
            out = self.block_3(x)
            x = torch.cat((x, out), 1)
            out = self.block_4(x)
        else:
            out_1 = self.block_1(x)
            out_2 = self.block_2(out_1)
            x_inter = self.addition.add(out_1, out_2)
            out_3 = self.block_3(x_inter)
            x_inter = self.addition.add(x_inter, out_3)
            x = torch.cat((x, x_inter), 1)
            out = self.block_4(x)
            
        if self.maxpool_bool:
            out = self.maxpool(out)
        return out
        
class ConvLayer(nn.Module):
    """
    A simple convolution layer with a norm layer and a non-linear activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, maxpool_bool=False, activation="gelu", **kwargs):
        super().__init__()
        self.activation = {"elu": nn.ELU(inplace=True), "relu": nn.ReLU(inplace=True), "gelu": nn.GELU()}[activation]
        self.norm_layer = nn.GroupNorm
        n_groups = out_channels // args.groupnorm_divisor
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                            stride=stride, padding=padding, bias=False), # TODO: remove bias
            self.norm_layer(n_groups, out_channels),
            self.activation,
            )
        
        self.maxpool_bool = maxpool_bool
        if maxpool_bool:
            self.maxpool = nn.MaxPool2d(2)
        
        self.apply(weights_init_kaiming)

    def forward(self, x):
    
        x = self.model(x)
        if self.maxpool_bool:
            x = self.maxpool(x)
        return x

        
class Decoder(nn.Module):
    """
    A simple upsampling layer, further processed with a convolutional block of choice.
    """
    def __init__(self, in_channels, out_channels, mid_channels=128, dense=False, skip_size=None, as_final_block=False, block=ShortResBlock, **kwargs):
        super().__init__()
        self.out_channels = out_channels
        self.dense = dense
        self.incoming_skip = skip_size is not None
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic')
        # self.upsample = UpProjModule(num_channels, num_channels)
        if self.incoming_skip:       
            self.conv = block(in_channels + skip_size, out_channels, mid_channels=mid_channels, dense=dense, maxpool_bool=False, as_final_block=as_final_block, **kwargs)
        else:
            self.conv = block(in_channels, out_channels, mid_channels=mid_channels, dense=dense, maxpool_bool=False, as_final_block=as_final_block, **kwargs)
            
        
    def forward(self, x, skip=None):
        
        x = self.upsample(x)
        if self.incoming_skip:
            assert skip is not None
            x = torch.cat((x, skip), dim=1) 
        out = self.conv(x)
      
        return out

class Acti(nn.Module):
    
    def __init__(self, input, output, activ_fuction=nn.Sigmoid):
        super().__init__()
        self.acti_func = activ_fuction()
        self.conv_2 = nn.Conv2d(input, output, kernel_size=3, padding=1, bias=True)
     
        
        self.apply(weights_init_kaiming)
        
    def forward(self, x):
        x = self.acti_func(x)
        x = self.conv_2(x)
        return x 

class Depth_Activation(nn.Module):
    """
    Create a depth map, by using a sigmoid activation, and then a linear convolution, for fine scaling and stretching.
    """
    def __init__(self, input, output, activ_fuction=nn.Sigmoid):
        super().__init__()
        iter_channel = 32
        self.acti_func = activ_fuction()
        self.conv_1 = nn.Conv2d(input, iter_channel, kernel_size=3, padding=1, bias=True)
        self.conv_2 = nn.Conv2d(iter_channel * 1, output, kernel_size=3, padding=1, bias=True)
        
    def forward(self, x):
        x_inter = self.conv_1(x)
        x_sigmoid = self.acti_func(x_inter)
        x = self.conv_2(x_sigmoid)
        return x 


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
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
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
    elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
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
        
        
################################ Helper Functions ################################

def create_tqdm_bar(iterable, desc):
    """
    Creates a nice looking progress bar.
    """
    return tqdm(enumerate(iterable),total=len(iterable), ncols=170, desc=desc)

def load_without_module_state_dict(model, state_dict):
    """
    Load state dict without the 'module' prefix
    """
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

def load_checkpoint_with_shape_match(model, checkpoint_dict):
    """
    1. Load state dict without the 'module' prefix (If trained earlier with some distributed version)
    2. Load only matching layers from the checkpoit, while notifying the use about the mismatching layers.
    """
    checkpoint = {k.replace('module.', ''): v for k, v in checkpoint_dict.items()}
    model_state_dict = model.state_dict()
    new_state_dict = {}
    # Iterate over the keys in the checkpoint
    for key in model_state_dict.keys():
        if key in checkpoint and checkpoint[key].shape == model_state_dict[key].shape:
            new_state_dict[key] = checkpoint[key]
        else:
            if key not in checkpoint:
                print(f"{args.hashtags_prefix} Key not in checkpoint: ", key)
            else:
                print(f"{args.hashtags_prefix} Shape mismatch: ", key, checkpoint[key].shape, model_state_dict[key].shape)
            new_state_dict[key] = model_state_dict[key]
    model.load_state_dict(new_state_dict, strict=True)