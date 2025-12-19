import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid = max(1, in_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c) 

        avg_out = self.mlp(avg)
        max_out = self.mlp(mx)
        out = avg_out + max_out
        out = self.sigmoid(out).view(b, c, 1, 1)
        return x * out.expand_as(x)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.pool_conv(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, use_ca=True):
        super().__init__()
        self.use_ca = use_ca
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

        if self.use_ca:
            self.ca = None

    def forward(self, x_dec, x_enc):
        x = self.up(x_dec)

        diff_h = x_enc.size(2) - x.size(2)
        diff_w = x_enc.size(3) - x.size(3)
        if diff_h != 0 or diff_w != 0:
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                          diff_h // 2, diff_h - diff_h // 2])

        if self.use_ca:
            if (self.ca is None) or (self.ca and self.ca_in_channels != x_enc.size(1)):
                self.ca_in_channels = x_enc.size(1)
                self.ca = ChannelAttention(self.ca_in_channels)

            x_enc = self.ca(x_enc)

        x = torch.cat([x_enc, x], dim=1)
        x = self.conv(x)
        return x

class UNetWithCA(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_c=64, bilinear=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_c = base_c

        self.inc = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c*2)             
        self.down2 = Down(base_c*2, base_c*4)         
        self.down3 = Down(base_c*4, base_c*8)        
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c*8, base_c*16 // factor)   

        self.bottleneck = DoubleConv(base_c*16 // factor, base_c*16 // factor)

        self.up1 = Up(base_c*16 // factor + base_c*8, base_c*8 // factor, bilinear=bilinear, use_ca=True)
        self.up2 = Up(base_c*8 // factor + base_c*4, base_c*4 // factor, bilinear=bilinear, use_ca=True)
        self.up3 = Up(base_c*4 // factor + base_c*2, base_c*2 // factor, bilinear=bilinear, use_ca=True)
        self.up4 = Up(base_c*2 // factor + base_c, base_c, bilinear=bilinear, use_ca=True)

        self.outc = nn.Conv2d(base_c, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)     
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        b = self.bottleneck(x5)
        d1 = self.up1(b, x4)
        d2 = self.up2(d1, x3)
        d3 = self.up3(d2, x2)
        d4 = self.up4(d3, x1)

        out = self.outc(d4)
        return out

def unet05res(in_channels, **kwargs):
    return UNetWithCA(in_channels, w=0.5, **kwargs)

def unet025res(in_channels, **kwargs):
    return UNetWithCA(in_channels, w=0.25, **kwargs)

def unet1res(in_channels, **kwargs):
    return UNetWithCA(in_channels, w=1, **kwargs)

def unet2res(in_channels, **kwargs):
    return UNetWithCA(in_channels, w=2, **kwargs)

def unet4res(in_channels, **kwargs):
    return UNetWithCA(in_channels, w=4, **kwargs)
