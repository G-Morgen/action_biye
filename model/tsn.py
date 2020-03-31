import torch
import torch.nn as nn
import math

import sys
try:
    sys.path.append('.')
    from model import resnet
    from model.utils import basic_ops
    from model import bn_inception
except:
    import resnet
    import bn_inception
    from utils import basic_ops


class tsn_model(nn.Module):

    def __init__(self, model, modality='rgb', inp=3, num_classes=150, input_size=224, input_segments = 8,dropout=0.5):
        super(tsn_model, self).__init__()

        if modality == 'flow':
            inp = 10
            
        self.num_classes = num_classes
        self.inp = inp
        self.input_segments = input_segments
        self._enable_pbn = False
        if model == 'resnet18':
            self.model = resnet.resnet18(inp=inp,pretrained=True)
        elif model == 'resnet34':
            self.model = resnet.resnet34(inp=inp,pretrained=True)
        elif model == 'resnet50':
            self.model = resnet.resnet50(inp=inp,pretrained=True)
        elif model == 'resnet101':
            self.model = resnet.resnet101(inp=inp,pretrained=True)
        elif model == 'bn_inception':
            self.model = bn_inception.bninception(inp=inp)

        self.modality = modality
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=dropout)
        in_channels = self.model.fc.in_features
        self.model.fc = None
        self.fc = nn.Linear(in_channels, num_classes)
        self.consensus = basic_ops.ConsensusModule('avg')

    def partialBN(self, enable):
        self._enable_pbn = enable

    def forward(self,x):
        
        b = x.size(0)
        x = x.view((-1, self.inp) + x.size()[-2:])

        x = self.model.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = x.view(b,-1,self.num_classes)
        # x = self.consensus(x)
        x = x.mean(dim=1)

        return x

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(tsn_model, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
                        # m.track_running_stats = False

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
                  
            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

def run_check_new_bn_inception():
    model = tsn_model('bn_inception',modality='rgb',num_classes=101)
    ckpt = "/4T/zhujian/ckpt/model/new_ucf101_rgb.pth"
    print('loading new ucf101 rgb ')
    a = torch.load(ckpt)
    # print(list(model.modules()))
    model.load_state_dict(a)

if __name__ == "__main__":
    
    run_check_new_bn_inception()
            

