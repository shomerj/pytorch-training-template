import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def normalize(in_channels, type='batchNorm'):
    if type == 'groupNorm':
        return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif type == 'batchNorm':
        return nn.BatchNorm2d(in_channels)

def nonlinearity(x, type='relu'):
    if type=='swish':
        return x*torch.sigmoid(x)
    elif type=='relu':
        return F.relu(x)

class ResnetBlock(nn.Module):
    def __init__(self,in_channels,
                      out_channels=None,
                      norm_type='batchNorm',
                      nonlinearity='relu',
                      expansion=1,
                      downsample=None,
                      dilation=1,
                      stride=1,
                          ):

        super(ResnetBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm_type = norm_type
        self.nonlinearity = nonlinearity
        self.stride = stride
        self.dilation = dilation
        self.downsample = downsample
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size=3,
                                stride=self.stride,
                                padding=1,
                                dilation=self.dilation)
        self.norm1 = normalize(out_channels, self.norm_type)
        self.conv2 = nn.Conv2d(out_channels,
                                out_channels * self.expansion,
                                kernel_size=3,
                                stride=1,
                                padding=1
                                )
        self.norm2 = normalize(out_channels * self.expansion)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = nonlinearity(out, self.nonlinearity)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = nonlinearity(out, self.nonlinearity)
        return out


class Model(nn.Module):
    def __init__(self,
                    num_classes,
                    layers,
                    planes_per_layer,
                    inplanes=32,
                    input_size=300,
                    norm_type='batchNorm',
                    nonlinearity='swish',
                    downsample_type='conv',
                    inblock_expansion=1,
                    downsample_input=True,
                    export=False
                    ):
        super(Model, self).__init__()

        self.num_classes = num_classes
        self.layers = layers
        self.input_size = input_size
        self.inplanes = inplanes
        self.norm_type = norm_type
        self.nonlinearity = nonlinearity
        self.planes_per_layer = planes_per_layer
        self.inblock_expansion = inblock_expansion
        self.downsample_input = downsample_input
        self.downsample_type = downsample_type
        self.stride = 2
        self.export = export

        if self.downsample_type == 'conv':
            self.downsample_by_pool = False
        elif self.downsample_type == 'pool':
            self.downsample_by_pool = True
        else:
            assert False, "Downsample type must be pool or conv"

        assert len(self.layers) == len(planes_per_layer), "Layers and planes per layer must be the same"

        self.layer1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.norm1 = normalize(self.inplanes, self.norm_type)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.MaxPool2d(2,2, ceil_mode=True)
        self.output_size = self.calculat_output_size()
        self.out_channels = planes_per_layer[-1] * self.inblock_expansion
        print(f'Number of output channels: {self.out_channels} | Size of resblock output {self.output_size}')
        self.output_layer = nn.Softmax(dim=1)
        self.linear_layer = nn.Linear(self.out_channels*self.output_size*self.output_size, 128)
        self.prediction_head = nn.Linear(self.out_channels*self.output_size*self.output_size, num_classes)
        self.sigmoid = nn.Sigmoid()
        layer_dict = OrderedDict()
        for i, block in enumerate(layers):
            planes = planes_per_layer[i]
            layer_dict[f'layer_{i+1}'] = self.make_layer(planes, block, self.stride, self.norm_type, self.downsample_by_pool)
        self.resnet_blocks = nn.Sequential(layer_dict)

    def calculat_output_size(self):
        div = 2**(len(self.layers)+1)
        if self.downsample_input:
            div *= 4
        out = self.input_size / div
        return math.ceil(out)

    def make_layer(self, planes, blocks, stride, norm_type, downsample_by_pool=False):
        layers = []
        downsample = None
        if stride != 1 or self.inplanes != planes*self.inblock_expansion:
            downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, planes*self.inblock_expansion,
                                kernel_size=1, stride=1 if downsample_by_pool else 2, padding=0),
                        normalize(planes*self.inblock_expansion, norm_type)
                        )
            if downsample_by_pool:
                downsample.add_module('pool', nn.MaxPool2d(2,2, ceil_mode=True))
        layers.append(ResnetBlock(self.inplanes, planes*self.inblock_expansion, norm_type, stride=stride, downsample=downsample))
        self.inplanes = planes * self.inblock_expansion
        for _ in range(1, blocks):
            layers.append(ResnetBlock(self.inplanes, planes*self.inblock_expansion, norm_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.norm1(out)
        out = nonlinearity(out, self.nonlinearity)
        if self.downsample_input:
            out = self.pool1(out)
        out = self.resnet_blocks(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), self.output_size*self.output_size*self.out_channels)
        out = self.linear_layer(out)
        out = nonlinearity(out, self.nonlinearity)
        out = self.prediction_head(out)
        if self.export:
            out = self.sigmoid(out)

        return out

def build_model(cfg):
    if type(cfg) == str:
        cfg = json.load(open(cfg))

    model_config = cfg['model']
    num_classes = model_config['num_classes']
    layers = model_config['layers']
    planes_per_layer = model_config["planes_per_layer"]
    inplanes = model_config['inplanes']
    norm_type = model_config['norm_type']
    nonlinearity = model_config['nonlinearity']
    downsample_type = model_config['downsample_type']
    input_size = cfg['data_loader']['args']['input_size']
    inblock_expansion = model_config["inblock_expansion"]
    downsample_input = model_config["downsample_input"]
    export = model_config['export']

    model = Model(num_classes,
                    layers,
                    planes_per_layer,
                    inplanes,
                    input_size[0],
                    norm_type,
                    nonlinearity,
                    downsample_type,
                    inblock_expansion,
                    downsample_input,
                    export
                    )
    return model
