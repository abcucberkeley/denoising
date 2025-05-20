# ========================================
# Custom Dilated Residual UNET (DRUNET) Model
# ========================================

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F


# Define the Standard Block of DRUNET
class StandardBlock(nn.Module):
    def __init__(self, in_channels, out_channels_1, out_channels_2, dropout=0.0, dilation=1):
        super(StandardBlock, self).__init__()
        self.doubleconv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels_1, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=True),
            nn.BatchNorm3d(out_channels_1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_channels_1, out_channels_2, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=True),
            nn.BatchNorm3d(out_channels_2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.doubleconv(x)


# Define the Residual Block of DRUNET
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels_1, out_channels_2, dropout=0.0, dilation=1):
        super(ResidualBlock, self).__init__()
        self.doubleconv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels_1, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=True),
            nn.BatchNorm3d(out_channels_1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_channels_1, out_channels_2, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=True),
            nn.BatchNorm3d(out_channels_2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.resconv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels_2, kernel_size=1, dilation=dilation, bias=True),
            nn.BatchNorm3d(out_channels_2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # print(f"X: {x.size()}")
        # print(f"X_dc: {self.doubleconv(x).size()}")
        # print(f"X_rc: {self.resconv(x).size()}")
        return torch.add(self.doubleconv(x), self.resconv(x))
    

# Define DRUNET
class DRUNET(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dropout=0.0, dilations=[1, 2, 4, 8], features=[16, 16, 16]):
        super(DRUNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.downs.append(StandardBlock(in_channels, features[0], features[0], dropout, dilations[0]))
        in_channels = features[0]
        for f in range(1, len(features)):
            self.downs.append(ResidualBlock(in_channels, features[f], features[f], dropout, dilations[f]))
            in_channels = features[f]

        for f in range(len(features)-1, 0, -1):
            self.ups.append(nn.ConvTranspose3d(features[f], features[f], kernel_size=2, stride=2))
            self.ups.append(ResidualBlock(features[f]*2, features[f], features[f]//2, dropout, dilations[f]))
            # self.ups.append(ResidualBlock(features[f] * 2, features[f], features[f], dropout, dilations[f]))
        self.ups.append(nn.ConvTranspose3d(features[0], features[0], kernel_size=2, stride=2))
        self.ups.append(StandardBlock(features[0] * 2, features[0], features[0], dropout, dilations[0]))

        self.bottleneck = ResidualBlock(features[-1], features[-1]*2, features[-1], dropout, dilations[-1])
        # self.bottleneck = ResidualBlock(features[-1], features[-1], features[-1], dropout, dilations[-1])
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        initial = x

        skip_connections = []

        # Contracting path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Expansive path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
                
            concat_skip = torch.cat((skip_connection, x), dim=1)  # Concatenate the skip connections
            x = self.ups[idx+1](concat_skip)
            
        x = self.final_conv(x)  # Final 1x1 convolution
        
        x = torch.add(initial, x)  # Final output added with input similar to CARE

        return x


if __name__ == "__main__":
    pass
