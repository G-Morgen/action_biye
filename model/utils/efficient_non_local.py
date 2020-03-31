import torch
from torch import nn
from torch.nn import functional as F


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.global_cluster = torch.rand((1,self.inter_channels),requires_grad=True,dtype=torch.float32)
        nn.init.kaiming_normal_(self.global_cluster,mode='fan_out')


        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 1)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.kaiming_normal_(self.W.weight, mode='fan_out',nonlinearity='relu')
            nn.init.constant_(self.W.bias, 0)

        
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        c = torch.matmul(g_x, self.global_cluster.permute(1,0))

        global_relation = torch.sum(c,dim=1)
        global_relation = global_relation * global_relation / g_x.size()[-1]
        
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        phi_x = phi_x.permute(0,2,1)

        d_x = phi_x * c
        d_x = torch.sum(d_x, dim=1, keepdim=True)
        y = torch.matmul(c, d_x) / phi_x
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        

        return z
       


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None,  bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, 
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None,  bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, 
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None,  bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, 
                                              bn_layer=bn_layer)


if __name__ == '__main__':
    from torch.autograd import Variable
    import torch
    from thop import profile

    sub_sample = True
    bn_layer = True

    # img = Variable(torch.zeros(2, 100, 20))
    # net = NONLocalBlock1D(100,  bn_layer=bn_layer, inter_channels=50)
    # out = net(img)
    # flops, params = profile(net, inputs=(img,), verbose=False)
    # print(flops / 1e9, params / 1e6)
    # print(out.size())

    # img = Variable(torch.zeros(2, 100, 20, 20))
    # net = NONLocalBlock2D(100,  bn_layer=bn_layer, inter_channels=50)
    # out = net(img)
    # flops, params = profile(net, inputs=(img,), verbose=False)
    # print(flops / 1e9, params / 1e6)
    # print(out.size())

    img = Variable(torch.randn(2, 100, 10, 20, 20))
    net = NONLocalBlock3D(100, bn_layer=bn_layer, inter_channels=50)
    out = net(img)
    flops, params = profile(net, inputs=(img,), verbose=False)
    print(flops / 1e9, params / 1e6)
    print(out.size())        