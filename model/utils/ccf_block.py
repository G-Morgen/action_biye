import torch
from torch import nn
from torch.nn import functional as F

def cross_correlation_operation(x,y):
    b,c = x.size(0), x.size(1)
    x = x.view(b,c,-1).transpose(1,2)
    y = y.view(b,c,-1)

    x_y = torch.bmm(x,y)

    return x_y

class cross_correlation_block(nn.Module):

    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True):
        super(cross_correlation_block, self).__init__()
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(2)
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(2)
            bn = nn.BatchNorm1d

        self.g_rgb = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.g_flow = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W_rgb = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            self.W_flow = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W_rgb[1].weight, 1)
            nn.init.constant_(self.W_flow[1].weight, 1)
            nn.init.constant_(self.W_rgb[1].bias, 0)
            nn.init.constant_(self.W_flow[1].bias, 0)
        else:
            self.W_rgb = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            self.W_flow = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.kaiming_normal_(self.W_rgb.weight, mode='fan_out',nonlinearity='relu')
            nn.init.kaiming_normal_(self.W_flow.weight, mode='fan_out',nonlinearity='relu')
            nn.init.constant_(self.W_rgb.bias, 0)
            nn.init.constant_(self.W_flow.bias, 0)


    def forward(self, rgb, flow):
        batch_size = rgb.size(0)
        
        rgb_embeded = self.g_rgb(rgb)
        flow_embeded = self.g_flow(flow)

        rgb_flow = cross_correlation_operation(rgb_embeded, flow_embeded)

        flow_embeded = flow_embeded.view(batch_size, self.inter_channels, -1).permute(0,2,1)
        rgb_embeded  = rgb_embeded.view(batch_size, self.inter_channels, -1).permute(0,2,1)

        rgb_y = torch.bmm(torch.softmax(rgb_flow,dim=-1), flow_embeded)
        flow_y = torch.bmm(torch.softmax(rgb_flow.transpose(2,1),dim=-1),rgb_embeded)
        
        rgb_y = rgb_y.permute(0, 2, 1).contiguous()
        rgb_y = rgb_y.view(batch_size, self.inter_channels, *rgb.size()[2:])

        flow_y = flow_y.permute(0, 2, 1).contiguous()
        flow_y = flow_y.view(batch_size, self.inter_channels, *flow.size()[2:])

        rgb_out = self.W_rgb(rgb_y)
        flow_out = self.W_flow(flow_y)

        rgb_z = rgb_out + rgb
        flow_z = flow_out + flow

        return rgb_z, flow_z

def run_check_ccfblock():
    rgb = torch.rand(1,3,28,28)
    flow = torch.rand(1,3,28,28)

    ccfblock = cross_correlation_block(3,3,dimension=2)
    d,c = ccfblock(rgb,flow)
    print(d.shape)

if __name__ == "__main__":
    
    run_check_ccfblock()


