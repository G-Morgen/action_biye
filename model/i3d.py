import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torchvision
################
#
# Modified https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# Adds support for B x T x C x H x W video data
#
################


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed3d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, t=3):
    """ 3D convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(t,3,3), stride=(1,stride,stride),
                     padding=(1 if t == 3 else 0, 1, 1), bias=False)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_temporal=False):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride, t=3 if use_temporal else 1)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes,t=1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_temporal=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3 if use_temporal else 1,1,1),
                             padding=(1 if use_temporal else 0, 0, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1,3,3), stride=(1,stride,stride),
                               padding=(0,1,1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, modality='rgb', inp=3, num_classes=150, input_size=224, input_segments = 8,dropout=0.5):
        self.inplanes = 64
        self.modality = modality
        if self.modality == 'flow':
            inp = 2
        else:
            inp = 3
        self.inp = inp

        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(inp, 64, kernel_size=(5,7,7), stride=(2,2,2), padding=(2,3,3), 
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.input_size = input_size
        if input_size == 224:
            self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(2,2,2), padding=(0,1,1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2,1,1), stride=(2,1,1), padding=(0,0,0))

        self.layer1 = self._make_layer(block, 64, layers[0], t=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, t=3)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, t=3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, t=3)

        # probably need to adjust this based on input spatial size
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        nn.init.normal_(self.fc.weight, std=0.001)

    def _make_layer(self, block, planes, blocks, stride=1, t=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1,stride,stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_temporal=True))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_temporal=True if i % t == 0 else False))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x is BxTxCxHxW
        # spatio-temporal video data
        if len(x.size()) != 5:
            b = x.size(0)
            x = x.view((b, -1, self.inp) + x.size()[-2:]).transpose(1,2)
        b,c,t,h,w = x.size()
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.input_size == 224:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.pool2(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        #print(x.size())
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        # currently making dense, per-frame predictions
        x = self.fc(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        # ignore fc layer
        state_dict = {k:v for k,v in state_dict.items() if 'fc' not in k}
        md = self.state_dict()
        for k,v in state_dict.items():
            if 'conv' in k or 'downsample.0' in k:
                if isinstance(v, nn.Parameter):
                    v = v.data
                if self.inp != 3 and k == 'conv1.weight':
                    v = torch.mean(v, dim=1).unsqueeze(1).repeat(1, self.inp, 1, 1)
                # CxKxHxW -> CxKxTxHxW
                D = md[k].size(2)
                v = v.unsqueeze(2).repeat(1,1,D,1,1).mul_(1/D)
                state_dict[k] = v

        md.update(state_dict)
        super(ResNet, self).load_state_dict(md, strict)


    
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    print(model)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, mode='rgb', **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if mode == 'flow':
        model = ResNet(BasicBlock, [3, 4, 6, 3], inp=20, **kwargs)
    else:
        model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, mode='rgb', **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if mode == 'flow':
        model = ResNet(Bottleneck, [3, 4, 6, 3], inp=20, **kwargs)
    else:
        model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def run_check_resnet34():
    import numpy as np
    net = resnet50(pretrained=True,num_classes = 101)
    o_st = model_zoo.load_url(model_urls['resnet50'])
    print(net)


if __name__ == '__main__':
    run_check_resnet34()
    
   