import torch 
import torch.nn as nn 

class t_fusion(nn.Module):

    def __init__(self,keep_ratio=0.75):
        super(t_fusion,self).__init__()
        self.keep_ratio = keep_ratio

    def forward(self,x):
        '''
        x: [b,c,t,h,w]
        '''
        b,c,t,h,w = x.size()

        keep_channel = int(c * self.keep_ratio)
        ratio_channel = c - keep_channel
        out = x[:,keep_channel:]
        out = out.view(b,-1,h,w).view(b,t,ratio_channel,h,w).transpose(2,1)
        out = torch.cat([x[:,:keep_channel],out],dim=1)
        
        return out

if __name__ == "__main__":

    x = torch.arange(32*4)
    x = x.view(1,32,4,1,1)
    a = t_fusion()
    out = a(x)
    print(x.squeeze(-1).squeeze(-1))
    print(out.squeeze(-1).squeeze(-1))