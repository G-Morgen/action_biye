import torch
import torch.nn as  nn
import torch.nn.functional as F
import torchvision
from sklearn.cluster import k_means
import numpy as np 
from models import TSN

class DFL(TSN):

    def __init__(self,k=5, **kwarg):
        super(DFL, self).__init__(**kwarg, use_middle_feature=True)
        self.k = k
        # self.dropout = nn.Dropout()
        base_model = kwarg.get('base_model')
        if base_model == 'resnet50':
            res3_layer = 1024 * 1
            res3_feature = self.base_model.state_dict()['layer3.5.conv1.weight']
        elif base_model == 'resnet34':
            res3_layer = 256 * 1
        elif base_model == 'resnet18':
            res3_layer = 256 * 1
        
        for p in self.parameters():
            p.requires_grad = False 

        self.non_random = nn.Sequential(
            nn.Conv2d(res3_layer, k * self.num_classes, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(k*self.num_classes),
            nn.ReLU()
        )

        # if base_model == 'resnet50':
        #     a = res3_feature.data.mean(dim=0).repeat(k*self.num_classes,1,1,1)
        #     self.non_random.weight.data = torch.nn.Parameter(torch.FloatTensor(a))
            
        # Local-Stream
        self.gmp = nn.AdaptiveMaxPool2d((1,1))
        # self.gmp = nn.AdaptiveAvgPool2d((1,1))

        self.cls = nn.Conv2d(k * self.num_classes, self.num_classes, kernel_size=1, stride=1, padding=0)

        # Cross-Pool
        self.cross_channel_pool = nn.AvgPool1d(kernel_size=k,stride=k,padding=0)
    
    def forward(self, input):
        sample_len = (3 if self.modality == "RGB" or self.modality == 'rgb' else 2) * self.new_length

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        b = input.size(0)
        input = input.view((-1, sample_len) + input.size()[-2:])
        middle_feature, base_out = self.base_model.forward_middle_features(input)
        res3 = []
        for i in middle_feature[::-1]:
            if i.size(-1) == 14:
                res3_feature = i 
                break
        # res3_feature = torch.cat(res3,dim=1)

        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)
        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])

        # Local-Path
        x5 = self.non_random(res3_feature)
        x5 = self.gmp(x5)
        out1 = self.cls(x5)
        out1 = out1.view(b, -1, self.num_classes)
        out1 = torch.mean(out1,dim=1)
        # cross-pool
        x6 = x5.view(b, -1, self.k * self.num_classes)
        out2 = self.cross_channel_pool(x6)
        out2 = out2.view(b, -1, self.num_classes)
        out2 = torch.mean(out2,dim=1)

        # output = self.consensus(base_out)
        output = base_out.view(b,-1,self.num_classes)
        # x = self.consensus(x)
        output = output.mean(dim=1)

        return output.squeeze(1), out1, out2


if __name__ == "__main__":
    a = torch.rand(3, 3 * 3, 224, 224)    
    net = DFL(num_class=101,num_segments=3,modality='rgb',base_model='resnet50')
    a = filter(lambda p: p.requires_grad, net.parameters())
    print(len(list(a)))
    # print(net)
    # output, out1, out2, _ = net(a)
    # print(out1.shape)