import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

class MultiSourceUNet(nn.Module):
    """Multi-Source U-Net for plantation detection using satellite imagery.
    
    Architecture:
    - Separate encoders for Sentinel-1, Sentinel-2, and MODIS data
    - Feature fusion module to combine multi-source information
    - U-Net style decoder with skip connections
    
    Args:
        n_classes (int): Number of output classes (default: 1 for binary segmentation)
    
    Input:
        x (torch.Tensor): [B, 8, H, W] tensor containing:
            - Sentinel-2 (4 bands): RGB + NIR
            - Sentinel-1 (2 bands): VV + VH
            - MODIS (2 bands): NDVI + EVI
    """
    def __init__(self, n_classes=1):
        super().__init__()
        
        # Separate encoders for different data sources
        self.s2_encoder = self._create_encoder(4)  # Sentinel-2 (4 bands)
        self.s1_encoder = self._create_encoder(2)  # Sentinel-1 (2 bands)
        self.modis_encoder = self._create_encoder(2)  # MODIS (2 bands)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(512 * 3, 512, 1),  # Combine features from all sources
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder remains the same
        self.decoder5 = ConvBlock(512, 256)
        self.decoder4 = ConvBlock(256 + 256, 128)
        self.decoder3 = ConvBlock(128 + 128, 64)
        self.decoder2 = ConvBlock(64 + 64, 32)
        self.decoder1 = ConvBlock(32, 32)
        
        self.final = nn.Conv2d(32, n_classes, kernel_size=1)
        
    def _create_encoder(self, in_channels):
        """Create encoder for each data source"""
        resnet = resnet34(pretrained=True)
        
        # Modify first conv layer
        first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, 
                              stride=2, padding=3, bias=False)
        
        # If using pretrained weights and in_channels == 3
        if in_channels == 3:
            first_conv.weight.data = resnet.conv1.weight.data
        elif in_channels == 4:
            first_conv.weight.data[:, :3] = resnet.conv1.weight.data
            first_conv.weight.data[:, 3] = resnet.conv1.weight.data.mean(dim=1)
        
        encoder = nn.ModuleList([
            nn.Sequential(first_conv, resnet.bn1, resnet.relu),
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        ])
        
        return encoder
    
    def _encode_single_source(self, x, encoder):
        features = []
        for layer in encoder:
            x = layer(x)
            features.append(x)
        return features
    
    def forward(self, x):
        # Store input size for later use
        input_size = x.shape[-2:]  # Get height and width
        
        # Split input into different sources
        s2_input = x[:, :4]    # Sentinel-2 bands
        s1_input = x[:, 4:6]   # Sentinel-1 bands
        modis_input = x[:, 6:] # MODIS bands
        
        # Encode each source
        s2_features = self._encode_single_source(s2_input, self.s2_encoder)
        s1_features = self._encode_single_source(s1_input, self.s1_encoder)
        modis_features = self._encode_single_source(modis_input, self.modis_encoder)
        
        # Fuse features from all sources
        fused_features = self.fusion(torch.cat([
            s2_features[-1],
            s1_features[-1],
            modis_features[-1]
        ], dim=1))
        
        # Decoding
        dec5 = self.decoder5(F.interpolate(fused_features, scale_factor=2))
        dec4 = self.decoder4(torch.cat([dec5, s2_features[-2]], dim=1))
        dec4 = F.interpolate(dec4, scale_factor=2)
        dec3 = self.decoder3(torch.cat([dec4, s2_features[-3]], dim=1))
        dec3 = F.interpolate(dec3, scale_factor=2)
        dec2 = self.decoder2(torch.cat([dec3, s2_features[-4]], dim=1))
        dec2 = F.interpolate(dec2, scale_factor=2)
        dec1 = self.decoder1(dec2)
        
        # Final interpolation to match input size
        final = F.interpolate(dec1, size=input_size, mode='bilinear', align_corners=False)
        
        return torch.sigmoid(self.final(final))

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()
        
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        loss = (1 - ((2. * intersection + self.smooth) /
                     (pred.sum(dim=2).sum(dim=2) + 
                      target.sum(dim=2).sum(dim=2) + self.smooth)))
        
        return loss.mean()

def test_model():
    """Test function to validate the MultiSourceUNet model and DiceLoss"""
    # Test parameters
    batch_size = 2
    height = 256
    width = 256
    n_classes = 1
    
    # Create dummy input data
    # 8 total bands: 4 (Sentinel-2) + 2 (Sentinel-1) + 2 (MODIS)
    x = torch.randn(batch_size, 8, height, width)
    
    # Create dummy target data (binary masks)
    target = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    
    # Initialize model
    model = MultiSourceUNet(n_classes=n_classes)
    
    # Test forward pass
    try:
        output = model(x)
        print(f"Forward pass successful!")
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected shape: {(batch_size, n_classes, height, width)}")
        
        # Check if dimensions match before assertion
        if output.shape != (batch_size, n_classes, height, width):
            print("Shape mismatch details:")
            print(f"Batch size: expected {batch_size}, got {output.shape[0]}")
            print(f"Channels: expected {n_classes}, got {output.shape[1]}")
            print(f"Height: expected {height}, got {output.shape[2]}")
            print(f"Width: expected {width}, got {output.shape[3]}")
        
        assert output.shape == (batch_size, n_classes, height, width)
        print("Output dimensions are correct!")
        
        # Test loss function
        criterion = DiceLoss()
        loss = criterion(output, target)
        print(f"Loss calculation successful! Loss value: {loss.item():.4f}")
        
        # Test backward pass
        loss.backward()
        print("Backward pass successful!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    test_model() 