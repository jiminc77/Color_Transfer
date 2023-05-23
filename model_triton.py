import torch
import torch.nn as nn
import numpy as np

class WaveEncoder(nn.Module):
    def __init__(self):
        super(WaveEncoder, self).__init__()

        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)

        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.pool1 = self.get_wav(64)  # Changed this

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.pool2 = self.get_wav(128)  # Changed this

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.pool3 = self.get_wav(256)  # Changed this

        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 0)
        
    def get_wav(self, in_channels, pool=True):
        """wavelet decomposition using conv2d"""
        harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
        harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
        harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

        harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
        harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
        harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
        harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

        filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
        filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
        filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
        filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

        if pool:
            net = nn.Conv2d
        else:
            net = nn.ConvTranspose2d

        LL = net(in_channels, in_channels,
                kernel_size=2, stride=2, padding=0, bias=False,
                groups=in_channels)
        LH = net(in_channels, in_channels,
                kernel_size=2, stride=2, padding=0, bias=False,
                groups=in_channels)
        HL = net(in_channels, in_channels,
                kernel_size=2, stride=2, padding=0, bias=False,
                groups=in_channels)
        HH = net(in_channels, in_channels,
                kernel_size=2, stride=2, padding=0, bias=False,
                groups=in_channels)

        LL.weight.requires_grad = False
        LH.weight.requires_grad = False
        HL.weight.requires_grad = False
        HH.weight.requires_grad = False

        LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
        LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
        HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
        HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()

        return LL, LH, HL, HH

    def forward(self, x):
        skips = {}
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skips, level)
        return x

    def encode(self, x, skips, level):
        assert level in {1, 2, 3, 4}

        if level == 1:
            out = self.conv0(x)
            out = self.relu(self.conv1_1(self.pad(out)))
            return out

        elif level == 2:
            out = self.relu(self.conv1_2(self.pad(x)))
            skips['conv1_2'] = out
            LL, LH, HL, HH = self.pool1(out)
            skips['pool1'] = [LH, HL, HH]
            out = self.relu(self.conv2_1(self.pad(LL)))
            return out

        elif level == 3:
            out = self.relu(self.conv2_2(self.pad(x)))
            skips['conv2_2'] = out
            LL, LH, HL, HH = self.pool2(out)
            skips['pool2'] = [LH, HL, HH]
            out = self.relu(self.conv3_1(self.pad(LL)))
            return out

        else:
            out = self.relu(self.conv3_2(self.pad(x)))
            out = self.relu(self.conv3_3(self.pad(out)))
            out = self.relu(self.conv3_4(self.pad(out)))
            skips['conv3_4'] = out
            LL, LH, HL, HH = self.pool3(out)
            skips['pool3'] = [LH, HL, HH]
            out = self.relu(self.conv4_1(self.pad(LL)))
            return out


class WaveDecoder(nn.Module):
    def __init__(self):
        super(WaveDecoder, self).__init__()
        multiply_in = 5

        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.conv4_1 = nn.Conv2d(512, 256, 3, 1, 0)

        self.LL3, self.LH3, self.HL3, self.HH3 = self.get_wav(256, pool=False)
        self.conv3_4_2 = nn.Conv2d(256*multiply_in, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_1 = nn.Conv2d(256, 128, 3, 1, 0)

        self.LL2, self.LH2, self.HL2, self.HH2 = self.get_wav(128, pool=False)
        self.conv2_2 = nn.Conv2d(128*multiply_in, 128, 3, 1, 0)
        self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)

        self.LL1, self.LH1, self.HL1, self.HH1 = self.get_wav(64, pool=False)
        self.conv1_2 = nn.Conv2d(64*multiply_in, 64, 3, 1, 0)
        self.conv1_1 = nn.Conv2d(64, 3, 3, 1, 0)
        
        self.recon_block3 = self.get_wav(256, pool=False)  #<-- change
        self.recon_block2 = self.get_wav(128, pool=False)  #<-- change
        self.recon_block1 = self.get_wav(64, pool=False)   #<-- change
        
    def get_wav(self, in_channels, pool=True):
        """wavelet decomposition using conv2d"""
        harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
        harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
        harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

        harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
        harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
        harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
        harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

        filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
        filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
        filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
        filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

        if pool:
            net = nn.Conv2d
        else:
            net = nn.ConvTranspose2d

        LL = net(in_channels, in_channels,
                kernel_size=2, stride=2, padding=0, bias=False,
                groups=in_channels)
        LH = net(in_channels, in_channels,
                kernel_size=2, stride=2, padding=0, bias=False,
                groups=in_channels)
        HL = net(in_channels, in_channels,
                kernel_size=2, stride=2, padding=0, bias=False,
                groups=in_channels)
        HH = net(in_channels, in_channels,
                kernel_size=2, stride=2, padding=0, bias=False,
                groups=in_channels)

        LL.weight.requires_grad = False
        LH.weight.requires_grad = False
        HL.weight.requires_grad = False
        HH.weight.requires_grad = False

        LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
        LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
        HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
        HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()

        return LL, LH, HL, HH

    def forward(self, x, skips):
        for level in [4, 3, 2, 1]:
            x = self.decode(x, skips, level)
        return x

    def decode(self, x, skips, level):
        assert level in {4, 3, 2, 1}
        if level == 4:
            out = self.relu(self.conv4_1(self.pad(x)))
            LH, HL, HH = skips['pool3']
            out = self.recon_block3(out, LH, HL, HH) 
            _conv3_4 = self.conv3_4_2
            out = self.relu(_conv3_4(self.pad(out)))
            out = self.relu(self.conv3_3(self.pad(out)))
            return self.relu(self.conv3_2(self.pad(out)))
        elif level == 3:
            out = self.relu(self.conv3_1(self.pad(x)))
            LH, HL, HH = skips['pool2']
            out = self.recon_block2(out, LH, HL, HH) 
            _conv2_2 = self.conv2_2_2
            return self.relu(_conv2_2(self.pad(out)))
        elif level == 2:
            out = self.relu(self.conv2_1(self.pad(x)))
            LH, HL, HH = skips['pool1']
            out = self.recon_block1(out, LH, HL, HH) 
            _conv1_2 = self.conv1_2_2
            return self.relu(_conv1_2(self.pad(out)))
        else:
            out = self.relu(self.conv1_1(self.pad(x)))
            return out