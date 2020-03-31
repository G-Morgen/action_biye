import torch
import torch.nn as nn
import math

from model.utils import ccf_block
import get_model
from model import resnet

class ccfnet(nn.Module):

    def __init__(self, model, num_classes=150, input_size=224, input_segments = 8,dropout=0.5, frozen=True):
        super(ccfnet, self).__init__()

        self.model = model
        self.frozen = frozen
        if model == 'resnet50':
            self.rgb_model =  get_model.get_model('resnet50',modality='rgb',num_classes=num_classes, input_size=input_size,
                input_segments=input_segments, dropout=0.8, use_middle_feature=True)
            self.flow_model =  get_model.get_model('resnet50',modality='flow',num_classes=num_classes, input_size=input_size,
                input_segments=input_segments, dropout=0.8, use_middle_feature=True)
            inp_features = [512 * 2, 1024 * 3, 2048 * 2]
        elif model == 'resnet101':
            self.rgb_model =  get_model.get_model('resnet101',modality='rgb',num_classes=num_classes, input_size=input_size,
                input_segments=input_segments, dropout=0.8, use_middle_feature=True)
            self.flow_model =  get_model.get_model('resnet101',modality='flow',num_classes=num_classes, input_size=input_size,
                input_segments=input_segments, dropout=0.8, use_middle_feature=True)
            inp_features = [512 * 2, 1024 * 12, 2048 * 2]
        elif model == 'bn_inception':
            self.rgb_model =  get_model.get_model('bn_inception',modality='rgb',num_classes=num_classes, input_size=input_size,
                input_segments=input_segments, dropout=0.8, use_middle_feature=True)
            self.flow_model =  get_model.get_model('bn_inception',modality='flow',num_classes=num_classes, input_size=input_size,
                input_segments=input_segments, dropout=0.8, use_middle_feature=True)
            inp_features = [256+320, 576+576+576+608+608, 1056+1024+1024]
        else:
            raise ValueError('no that model architecture')
        
        self.frozen_rgb_flow(frozen)

        self.ccf = ccf(
            inp_features=inp_features,
            input_segments=input_segments,
            refine_blocks=[2,2,2],
            num_classes=num_classes,
            dropout=dropout)

    def frozen_rgb_flow(self, flag=False):

        if flag:
            for p in self.parameters():
                p.requires_grad = False 
            self.rgb_model.eval()
            self.flow_model.eval()

    def get_every_other_feature(self, l):
        out = []
        for idx, i in enumerate(l):
            if idx % 2 == 0:
                out.append(i)
        return torch.cat(out, dim=1)

    def get_rgb_flow_feature_groups(self, r_m, f_m):
        r_28 = []; f_28 = []
        r_14 = []; f_14 = []
        r_7  = []; f_7  = []
        
        for r, f in zip(r_m,f_m):
            if r.size(-1) == 28:
                r_28.append(r)
                f_28.append(f)
            elif r.size(-1) == 14:
                r_14.append(r)
                f_14.append(f)
            elif r.size(-1) == 7:
                r_7.append(r)
                f_7.append(f)
        
        if self.model in ['resnet50','resnet101']:
            r_28_feature = self.get_every_other_feature(r_28)
            r_14_feature = self.get_every_other_feature(r_14)
            r_7_feature  = self.get_every_other_feature(r_7)
            f_28_feature = self.get_every_other_feature(f_28)
            f_14_feature = self.get_every_other_feature(f_14)
            f_7_feature  = self.get_every_other_feature(f_7)
        else:
            r_28_feature = torch.cat(r_28,dim=1)
            r_14_feature = torch.cat(r_14,dim=1)
            r_7_feature  = torch.cat(r_7,dim=1)
            f_28_feature = torch.cat(f_28,dim=1)
            f_14_feature = torch.cat(f_14,dim=1)
            f_7_feature  = torch.cat(f_7,dim=1)

        return [
            [r_28_feature, f_28_feature],
            [r_14_feature, f_14_feature],
            [r_7_feature , f_7_feature]
        ]


    def forward(self,r,f):
        r_m,r_out = self.rgb_model(r)
        f_m,f_out = self.flow_model(f)

        x = self.get_rgb_flow_feature_groups(r_m, f_m)

        fusion_out = self.ccf(x)
        
        return r_out, f_out, fusion_out

        

class ccf(nn.Module):

    def __init__(self, 
                inp_features = [128, 256, 512], 
                mid_features=256, 
                num_classes=101, 
                fusion_method='add',
                refine_blocks = [2,2,2],
                input_segments = 8,
                dropout=0.5):
        super(ccf, self).__init__()

        l = len(inp_features)
        self.mid_features = mid_features
        self.fusion_method = fusion_method
        self.inp_len = input_segments
        self.refine_blocks = refine_blocks
        self.num_classes = num_classes
        if fusion_method == 'add':
            self.inplanes = 256
        else:
            self.inplanes = 768

        self.rgb2embed = nn.ModuleList([nn.Conv2d(inp_features[i], mid_features, 1, bias=False) for i in range(l)])
        self.flow2embed = nn.ModuleList([nn.Conv2d(inp_features[i], mid_features, 1, bias=False) for i in range(l)])

        self.refine_layers = self._build_refine_layer(l)

        self.ccf_blocks = nn.ModuleList([ccf_block.cross_correlation_block(self.mid_features) for i in range(l)])
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Conv2d(self.mid_features, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feature_groups = x
        rf_fusion = None
        for idx, sample in enumerate(feature_groups):
            rgb, flow = sample
            rgb_emb = self.rgb2embed[idx](rgb)
            flow_emb = self.flow2embed[idx](flow)

            rgb_z, flow_z = self.ccf_blocks[idx](rgb_emb, flow_emb)
            if idx == 0:
                rf_fusion = rgb_z + flow_z
            else:
                rf_fusion = rgb_z + flow_z + rf_fusion
            rf_fusion = self.refine_layers[idx](rf_fusion)
            rf_fusion = self.dropout(rf_fusion)

        out = self.avg_pool(rf_fusion)
        out = self.dropout(out)
        out = self.fc(out)
        output = out.squeeze(-1).squeeze(-1).view(-1, self.inp_len, self.num_classes)
        output = output.mean(dim=1)
        return output.squeeze(1)

    def _build_refine_layer(self,l):
        first_refine = self._make_layer(resnet.Bottleneck, 256, 64, self.refine_blocks[0], stride=2)
        layers = [] 
        layers.append(first_refine)
        for i in range(1,l):
            layers.append(self._make_layer(resnet.Bottleneck, self.inplanes, 64, self.refine_blocks[i], stride=2 if i != l-1 else 1))
        
        return nn.ModuleList(layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    

def run_check_ccf():
    model = ccf(fusion_method='concat',inp_features=[3,3,3])
    a = torch.rand(1,3,28,28)
    b = torch.rand(1,3,28,28)
    c = torch.rand(1,3,14,14)
    d = torch.rand(1,3,14,14)
    e = torch.rand(1,3,7,7)
    f = torch.rand(1,3,7,7)
    x = [[a,b],[c,d],[f,e]]
    d = model(x)

def run_check_ccfnet():
    a = torch.rand(8,3 * 3,224,224)
    b = torch.rand(8,10 * 3,224,224)

    model = ccfnet('bn_inception', input_segments=3,frozen=True)
    c = filter(lambda y: y.requires_grad, model.parameters())
    
    
    gg,dd,cc = model(a,b)
    print(gg.shape, dd.shape, cc.shape)

if __name__ == "__main__":
    # run_check_cross_correlation_fusion()
    run_check_ccfnet()



