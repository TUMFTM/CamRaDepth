import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.getcwd())

import torch.nn as nn
import torch
import numpy as np
from torchinfo import summary

from utils.args import args
from models.simplified_attention import SimplifiedTransformer as SimpTR
from functools import partial
from utils.utils import Decoder, ShortResBlock, Seg_Block, Depth_Activation, ConvLayer


def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

class CamRaDepth(nn.Module):
    def __init__(
            self,
            img_size=(416, 800),
            heads=(1, 2, 4, 8),
            ff_expansion=(8, 8, 4, 4),
            reduction_ratio=(8, 4, 2, 1),
            depths=(3, 10, 16, 5),
            dims = (64, 128, 160, 256),
            input_channels=None,
            **kwargs
    ):
    
        super().__init__()
        
        # Hyperparameters
        self.depths = depths
        self.mid_channels = 128
        self.num_classes = args.num_classes
        self.dense = True
        self.dims = dims
        self.as_final_block = False
        self.unsupervised_seg = args.get("unsupervised_seg", False)
        self.supervised_seg = args.get("supervised_seg", False)
        self.img_size = np.array(img_size)
        input_channels = input_channels if input_channels is not None else args.input_channels
        
        dims, heads, ff_expansion, reduction_ratio, self.depths = map(partial(cast_tuple, depth = 4), (dims, heads, ff_expansion, reduction_ratio, self.depths))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, self.depths))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'
        assert input_channels > 0, 'input_channels must be > 0'
        
        # Architecture
        
        # Encoder
        self.dest_encoder = SimpTR(
            img_size=img_size, in_chans=input_channels, num_classes=self.num_classes,
            embed_dims=dims, num_heads=heads, mlp_ratios=ff_expansion, qkv_bias=True, qk_scale=None, drop_rate=0,
            drop_path_rate=0.1, attn_drop_rate=0., depths=self.depths, sr_ratios=reduction_ratio)
        
        
        conv_layer = ConvLayer
        self.from_encoder_1 = conv_layer(dims[-1], dims[-1], 1, padding=0)
        self.from_encoder_2 = conv_layer(dims[-2], dims[-2], 1, padding=0)
        self.from_encoder_3 = conv_layer(dims[-3], dims[-3], 1, padding=0)
        self.from_encoder_4 = conv_layer(dims[-4], dims[-4], 1, padding=0)

        # Depth
        self.depth_upsample = nn.ModuleList([
            Decoder(dims[-1], self.mid_channels, skip_size=dims[-2], dense=self.dense, as_final_block=self.as_final_block, block=ShortResBlock),
            Decoder(self.mid_channels, self.mid_channels, skip_size=dims[-3], dense=self.dense, as_final_block=self.as_final_block, block=ShortResBlock),
            Decoder(self.mid_channels, self.mid_channels, skip_size=dims[-4], dense=self.dense, as_final_block=self.as_final_block, block=ShortResBlock),
            Decoder(self.mid_channels + 1, self.mid_channels, dense=self.dense, as_final_block=self.as_final_block, block=ShortResBlock),
            Decoder(self.mid_channels + 1, self.mid_channels, skip_size=input_channels, dense=self.dense, as_final_block=self.as_final_block, block=ShortResBlock, mid_channels=128),
        ])
        
        self.depth_activation_3 = Depth_Activation(self.mid_channels, 1)
        self.depth_activation_4 = Depth_Activation(self.mid_channels + 1 * self.supervised_seg + 1 * self.unsupervised_seg, 1)
        self.depth_activation_5 = Depth_Activation(self.mid_channels + 1 * self.supervised_seg + 1 * self.unsupervised_seg, 1)
        
        # Seg
        if self.supervised_seg or self.unsupervised_seg:
            self.seg_upsample = nn.ModuleList([
                Decoder(self.mid_channels + 1, self.mid_channels, dense=self.dense, as_final_block=self.as_final_block, block=ShortResBlock),
                Decoder(self.mid_channels + 1 , self.mid_channels, skip_size=input_channels, dense=self.dense, as_final_block=self.as_final_block, block=ShortResBlock, mid_channels=128),
            ])
        
        if self.supervised_seg:
            self.seg_block = Seg_Block(self.num_classes)
            self.seg_conv_stage_4 = nn.Conv2d(self.mid_channels, self.num_classes, kernel_size=3, stride=1, padding=1)
            self.seg_conv_final = nn.Conv2d(self.mid_channels, self.num_classes, kernel_size=3, stride=1, padding=1)
        
        if self.unsupervised_seg:
            self.unsup_seg_block = Seg_Block(19)
            self.unsup_stage_4 = nn.Conv2d(self.mid_channels, 19, kernel_size=3, stride=1, padding=1)
            self.unsup_final = nn.Conv2d(self.mid_channels, 19, kernel_size=3, stride=1, padding=1)
            
        self.dropout = nn.Dropout2d(0.2)
        
    
    def dest_decoder(self, lay_out, x):
        
        unsup_map = None
        sup_seg_map = None
        seg_logits_final = None
        seg_map = None
        seg_features = None
        
        
        # Convolve the attention blocks, to be used in skip connections.
        encoded_1 = self.from_encoder_1(lay_out[-1]) # Size: [B, 320, 13, 25]
        encoded_2 = self.from_encoder_2(lay_out[-2]) # Size: [B, 224, 26, 50]
        encoded_3 = self.from_encoder_3(lay_out[-3]) # Size: [B, 128, 52, 100]
        encoded_4 = self.from_encoder_4(lay_out[-4]) # Size: [B, 64, 104, 200]
        
        # Perform upscaling, conctanation with the appropriate skip connection, and further convolution.
        decoder_stage_1 = self.dropout(self.depth_upsample[0](encoded_1, encoded_2)) # Size: [B, self.mid_channels, 26, 50]
        decoder_stage_2 = self.dropout(self.depth_upsample[1](decoder_stage_1, encoded_3)) # [B, self.mid_channels, 52, 100]
    
        decoder_stage_3 = self.dropout(self.depth_upsample[2](decoder_stage_2, encoded_4))
        inter_depth_3 = self.depth_activation_3(decoder_stage_3)         # Size: [B, 1, 104, 200]
        decoder_stage_3 = torch.cat([decoder_stage_3, inter_depth_3], 1) # Size: [B, self.mid_channels + 1, 104, 200]
        
        # With Seg stage 3-4  # 
        decoder_stage_4 = self.dropout(self.depth_upsample[3](decoder_stage_3)) # Size: [B, self.mid_channels, 208, 400]
        
        if self.supervised_seg or self.unsupervised_seg:
            seg_features = self.dropout(self.seg_upsample[0](decoder_stage_3))
        
        if self.supervised_seg:
            seg_logits_inter = self.seg_conv_stage_4(seg_features)
            sup_seg_map = self.seg_block(seg_logits_inter)
            seg_map = sup_seg_map
        
        if self.unsupervised_seg: 
            unsup_map = self.unsup_stage_4(seg_features)
            unsup_map = self.unsup_seg_block(unsup_map)
            seg_map = unsup_map if sup_seg_map is None else torch.cat([sup_seg_map, unsup_map], 1)
        
        if self.supervised_seg:
            seg_features = torch.cat((seg_features, sup_seg_map), dim=1)
        elif self.unsupervised_seg:
            seg_features = torch.cat((seg_features, unsup_map), dim=1)
            
        tmp = torch.cat((decoder_stage_4, seg_map), dim=1) if seg_map is not None else decoder_stage_4
            
        inter_depth_4 = self.depth_activation_4(tmp)        
        decoder_stage_4 = torch.cat([decoder_stage_4, inter_depth_4], 1)

        # Final predictions - last stage:
        decoder_stage_5 = self.dropout(self.depth_upsample[4](decoder_stage_4, x))
        
        if self.supervised_seg or self.unsupervised_seg:
            seg_features = self.dropout(self.seg_upsample[1](seg_features, x))
        
        if self.supervised_seg:
            seg_logits_final = self.seg_conv_final(seg_features)
            sup_seg_map = self.seg_block(seg_logits_final)
            seg_map = sup_seg_map
        
        if self.unsupervised_seg: 
            unsup_map = self.unsup_final(seg_features)
            unsup_map = self.unsup_seg_block(unsup_map)
            seg_map = unsup_map if sup_seg_map is None else torch.cat([sup_seg_map, unsup_map], 1)
        
        tmp = torch.cat((decoder_stage_5, seg_map), dim=1) if seg_map is not None else decoder_stage_5
        final_depth = self.depth_activation_5(tmp)   # Size: [B, 1, 416, 800]
        
        # Un comment the following line, in order for "summary" to not crash.
        # return final_depth 
        return {"depth": {"intermediate_depths": (None, None, inter_depth_3, inter_depth_4), "final_depth": final_depth}, 
                "seg": {"final_seg": seg_logits_final, "intermediate_seg": None, "unsup_map": unsup_map}}
         
     
    def forward(self, x):
        layer_outputs, _ = self.dest_encoder(x)
        ret_dict = self.dest_decoder(layer_outputs, x)
        return ret_dict

         
if __name__ == "__main__":
    summary(CamRaDepth(), (2, 7, 416, 800), device="cpu")
