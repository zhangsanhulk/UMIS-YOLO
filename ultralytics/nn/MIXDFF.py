import torch
import torch.nn as nn
import torch.nn.functional as F


class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)



class DFF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        

        self.conv_adj_x = None
        self.conv_adj_skip = None
        

        self.channel_enhance = nn.Sequential(
            nn.Conv2d(dim*2, dim*2, kernel_size=1),
            DyT(dim*2),  
            nn.ReLU(inplace=True),
        )
        

        self.att_conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.att_conv2 = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.nonlin = nn.Sigmoid()
        

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(dim*2, dim, kernel_size=1, bias=False),
            DyT(dim), 
            nn.ReLU(inplace=True)
        )
        

        self.mutiscale_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim//4, bias=False),
            DyT(dim),  
            nn.ReLU(inplace=True)
        )
        

        self.final_dy_t = DyT(dim)
        

        self.shortcut = nn.Sequential(
            nn.Conv2d(dim*2, dim, kernel_size=1, bias=False),
            DyT(dim)  
        )

    def forward(self, x1):
        skip, x = x1
        

        if x.shape[1] != self.dim:
            if self.conv_adj_x is None:
                self.conv_adj_x = nn.Conv2d(x.shape[1], self.dim, kernel_size=1, bias=False).to(x.device)
            x = self.conv_adj_x(x)
        if skip.shape[1] != self.dim:
            if self.conv_adj_skip is None:
                self.conv_adj_skip = nn.Conv2d(skip.shape[1], self.dim, kernel_size=1, bias=False).to(skip.device)
            skip = self.conv_adj_skip(skip)
        

        if skip.shape[2:] != x.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=True)
        

        channel_enhanced = self.channel_enhance(torch.cat([x, skip], dim=1))
        

        fused = self.fuse_conv(channel_enhanced)
        

        att1 = self.att_conv1(x)
        att2 = self.att_conv2(skip)
        att = self.nonlin(att1 + att2)
        

        multiscale = self.mutiscale_conv(fused * att)
        

        residual = self.shortcut(torch.cat([x, skip], dim=1))
        
    
        output = self.final_dy_t(multiscale + residual)
        
        return output


