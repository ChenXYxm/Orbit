from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import resnet
from base import BaseModel


class FCNet(BaseModel):
    """A fully-convolutional network with an encoder-decoder architecture.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.opt = {}
        self._in_channels = in_channels
        self._out_channels = out_channels
        
        
        self._encoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        "enc-conv0",
                        nn.Conv2d(
                            in_channels,
                            64,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    ("enc-norm0", nn.BatchNorm2d(64)),
                    ("enc-relu0", nn.ReLU(inplace=True)),
                    ("enc-pool1", nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
                    (
                        "enc-resn2",
                        resnet.BasicBlock(
                            64,
                            128,
                            downsample=nn.Sequential(
                                nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                nn.BatchNorm2d(128),
                            ),
                            dilation=1,
                        ),
                    ),
                    ("enc-pool3", nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
                    (
                        "enc-resn4",
                        resnet.BasicBlock(
                            128,
                            256,
                            downsample=nn.Sequential(
                                nn.Conv2d(128, 256, kernel_size=1, bias=False),
                                nn.BatchNorm2d(256),
                            ),
                            dilation=1,
                        ),
                    ),
                    
                ]
            )
        )
        self._encoder_stop = nn.Sequential(
            OrderedDict(
                [
                    (
                        "enc-conv0",
                        nn.Conv2d(
                            in_channels,
                            64,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    ("enc-norm0", nn.BatchNorm2d(64)),
                    ("enc-relu0", nn.ReLU(inplace=True)),
                    ("enc-pool1", nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
                    (
                        "enc-resn2",
                        resnet.BasicBlock(
                            64,
                            128,
                            downsample=nn.Sequential(
                                nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                nn.BatchNorm2d(128),
                            ),
                            dilation=1,
                        ),
                    ),
                    ("enc-pool3", nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
                    (
                        "enc-resn4",
                        resnet.BasicBlock(
                            128,
                            256,
                            downsample=nn.Sequential(
                                nn.Conv2d(128, 256, kernel_size=1, bias=False),
                                nn.BatchNorm2d(256),
                            ),
                            dilation=1,
                        ),
                    ),
                    
                ]
            )
        )
        self._decoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        "dec-resn1",
                        resnet.BasicBlock(
                            256,
                            128,
                            downsample=nn.Sequential(
                                nn.Conv2d(256, 128, kernel_size=1, bias=False),
                                nn.BatchNorm2d(128),
                            ),
                            dilation=1,
                        ),
                    ),
                    (
                        "dec-upsm2",
                        Interpolate(
                            scale_factor=2, mode="bilinear", align_corners=True
                        ),
                    ),
                    (
                        "dec-resn3",
                        resnet.BasicBlock(
                            128,
                            64,
                            downsample=nn.Sequential(
                                nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                nn.BatchNorm2d(64),
                            ),
                            dilation=1,
                        ),
                    ),
                    (
                        "dec-upsm4",
                        Interpolate(
                            scale_factor=2, mode="bilinear", align_corners=True
                        ),
                    ),
                    (
                        "dec-conv5",
                        nn.Conv2d(64, out_channels, kernel_size=1, stride=1, bias=True),
                    ),
                ]
            )
        )
        self._decoder_stop = nn.Sequential(
            OrderedDict(
                [
                    (
                        "dec-resn1",
                        resnet.BasicBlock(
                            256,
                            128,
                            downsample=nn.Sequential(
                                nn.Conv2d(256, 128, kernel_size=1, bias=False),
                                nn.BatchNorm2d(128),
                            ),
                            dilation=1,
                        ),
                    ),
                    (
                        "dec-upsm2",
                        Interpolate(
                            scale_factor=2, mode="bilinear", align_corners=True
                        ),
                    ),
                    (
                        "dec-resn3",
                        resnet.BasicBlock(
                            128,
                            64,
                            downsample=nn.Sequential(
                                nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                nn.BatchNorm2d(64),
                            ),
                            dilation=1,
                        ),
                    ),
                    (
                        "dec-upsm4",
                        Interpolate(
                            scale_factor=2, mode="bilinear", align_corners=True
                        ),
                    ),
                    (
                        "dec-conv5",
                        nn.Conv2d(64, out_channels, kernel_size=1, stride=1, bias=True),
                    ),
                ]
            )
        )
        self._init_weights()
        self.opt = optim.Adam(params=self.parameters(),lr=0.0001)
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_enc = self._encoder(x)
        out_dec = self._decoder(out_enc)

        out_enc_stop = self._encoder(x)
        out_dec_stop = self._decoder(out_enc_stop)
        return out_dec,out_dec_stop


class Interpolate(nn.Module):
    def __init__(
        self,
        size=None,
        scale_factor=None,
        mode="nearest",
        align_corners=None,
    ):
        super().__init__()

        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(
            input, self.size, self.scale_factor, self.mode, self.align_corners
        )