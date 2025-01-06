import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from einops import rearrange
from utils.utils import pkl_load
from .Decompose import Decompose
from einops.layers.torch import Rearrange
from .attn import DAC_structure, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding
from .RevIN import RevIN
from .dctnet import dct_channel_block
from timm.models.layers import trunc_normal_


class MYdetector(nn.Module):
    def __init__(self, win_size, enc_in, c_out, n_heads=1, d_model=256, e_layers=3, channel=55, k=3, dropout=0.0,
                 activation='gelu', output_attention=True, kernel_size=25):
        super(MYdetector, self).__init__()
        self.win_size = win_size
        self.layer = e_layers
        self.pred_len = 0
        self.seq_len = win_size

        # Leddam
        self.revin_layer = RevIN(channel=enc_in, output_dim=win_size)
        self.Decompose = Decompose(enc_in, win_size, d_model,
                             dropout, 'no', kernel_size=kernel_size, n_layers=e_layers)
        self.Linear_main = nn.Linear(d_model, win_size)
        self.Linear_res = nn.Linear(d_model, win_size)
        self.Linear_main.weight = nn.Parameter(
            (1 / d_model) * torch.ones([win_size, d_model]))
        self.Linear_res.weight = nn.Parameter(
            (1 / d_model) * torch.ones([win_size, d_model]))
        #self.dct_layer = dct_channel_block(win_size)
        #self.mfe = MFEblock(enc_in, [2, 4])  # channels, atrous_rates
        
        


    def forward(self, inp):
        inp = self.revin_layer(inp)
        #for i in range(self.layer):
        #    inp = self.mfe(inp.permute(0, 2, 1)).permute(0, 2, 1)
        
        res, main = self.Decompose(inp)
        main_out = self.Linear_main(main.permute(0, 2, 1)).permute(0, 2, 1)
        res_out = self.Linear_res(res.permute(0, 2, 1)).permute(0, 2, 1)
        pred = main_out + res_out
        pred = self.revin_layer.inverse_normalize(pred)
        return pred


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            # groups = in_channels
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class oneConv(nn.Module):
    # ����+ReLU����
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_sizes, padding=paddings, dilation=dilations,
                      bias=False),  ###, bias=False
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class MFEblock(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(MFEblock, self).__init__()
        out_channels = in_channels
        rate1, rate2 = tuple(atrous_rates)
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1, dilation=1, bias=False),
            # groups = in_channels , bias=False
            nn.BatchNorm1d(out_channels),
            nn.ReLU())
        self.layer2 = ASPPConv(in_channels, out_channels, rate1)
        self.layer3 = ASPPConv(in_channels, out_channels, rate2)
        #self.layer4 = ASPPConv(in_channels, out_channels, rate3)
        self.project = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(), )
        # nn.Dropout(0.5))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.softmax = nn.Softmax(dim=2)
        self.softmax_1 = nn.Sigmoid()
        self.SE1 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.SE2 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.SE3 = oneConv(in_channels, in_channels, 1, 0, 1)
        #self.SE4 = oneConv(in_channels, in_channels, 1, 0, 1)

    def forward(self, x):
        y0 = self.layer1(x)
        y1 = self.layer2(y0 + x)
        y2 = self.layer3(y1 + x)
        #y3 = self.layer4(y2 + x)
        # res = torch.cat([y0,y1,y2,y3], dim=1)
        y0_weight = self.SE1(self.gap(y0))
        y1_weight = self.SE2(self.gap(y1))
        y2_weight = self.SE3(self.gap(y2))
        #y3_weight = self.SE4(self.gap(y3))
        weight = torch.cat([y0_weight, y1_weight, y2_weight], 2)
        weight = self.softmax(self.softmax_1(weight))
        y0_weight = torch.unsqueeze(weight[:, :, 0], 2)
        y1_weight = torch.unsqueeze(weight[:, :, 1], 2)
        y2_weight = torch.unsqueeze(weight[:, :, 2], 2)
        #y3_weight = torch.unsqueeze(weight[:, :, 3], 2)
        x_att = y0_weight * y0 + y1_weight * y1 + y2_weight * y2 #+ y3_weight * y3
        return self.project(x_att + x)


