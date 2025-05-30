# Kaggle U²-Net Yaprak Segmentasyon Modeli
# Gerekli kütüphaneleri yükle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

# GPU kontrolü
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Kullanılan cihaz: {device}")

# Veri seti yapılandırması - Kaggle dizin yapınıza göre düzenleyin
IMAGE_DIR = "/kaggle/input/dataset/dataset/images"  # Renkli yaprak görselleri
MASK_DIR = "/kaggle/input/dataset/dataset/mask"    # Segmente edilmiş görsel maskeleri

# Model parametreleri
IMG_SIZE = 320
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50

# Basit konvolüsyon bloğu
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dirate=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

# RSU-7 bloğu
class RSU7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()
        
        self.rebnconvin = ConvBlock(in_ch, out_ch)
        
        self.rebnconv1 = ConvBlock(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv2 = ConvBlock(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv3 = ConvBlock(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv4 = ConvBlock(mid_ch, mid_ch)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv5 = ConvBlock(mid_ch, mid_ch)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv6 = ConvBlock(mid_ch, mid_ch)
        
        self.rebnconv7 = ConvBlock(mid_ch, mid_ch, dirate=2)
        
        self.rebnconv6d = ConvBlock(mid_ch*2, mid_ch)
        self.rebnconv5d = ConvBlock(mid_ch*2, mid_ch)
        self.rebnconv4d = ConvBlock(mid_ch*2, mid_ch)
        self.rebnconv3d = ConvBlock(mid_ch*2, mid_ch)
        self.rebnconv2d = ConvBlock(mid_ch*2, mid_ch)
        self.rebnconv1d = ConvBlock(mid_ch*2, out_ch)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        
        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)
        
        hx6 = self.rebnconv6(hx)
        
        hx7 = self.rebnconv7(hx6)
        
        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = F.interpolate(hx6d, size=(hx5.size(2), hx5.size(3)), mode='bilinear', align_corners=False)
        
        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=(hx4.size(2), hx4.size(3)), mode='bilinear', align_corners=False)
        
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=(hx3.size(2), hx3.size(3)), mode='bilinear', align_corners=False)
        
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=(hx2.size(2), hx2.size(3)), mode='bilinear', align_corners=False)
        
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=(hx1.size(2), hx1.size(3)), mode='bilinear', align_corners=False)
        
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        
        return hx1d + hxin

# RSU-6 bloğu
class RSU6(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()
        
        self.rebnconvin = ConvBlock(in_ch, out_ch)
        
        self.rebnconv1 = ConvBlock(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv2 = ConvBlock(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv3 = ConvBlock(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv4 = ConvBlock(mid_ch, mid_ch)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv5 = ConvBlock(mid_ch, mid_ch)
        
        self.rebnconv6 = ConvBlock(mid_ch, mid_ch, dirate=2)
        
        self.rebnconv5d = ConvBlock(mid_ch*2, mid_ch)
        self.rebnconv4d = ConvBlock(mid_ch*2, mid_ch)
        self.rebnconv3d = ConvBlock(mid_ch*2, mid_ch)
        self.rebnconv2d = ConvBlock(mid_ch*2, mid_ch)
        self.rebnconv1d = ConvBlock(mid_ch*2, out_ch)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        
        hx5 = self.rebnconv5(hx)
        
        hx6 = self.rebnconv6(hx5)
        
        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=(hx4.size(2), hx4.size(3)), mode='bilinear', align_corners=False)
        
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=(hx3.size(2), hx3.size(3)), mode='bilinear', align_corners=False)
        
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=(hx2.size(2), hx2.size(3)), mode='bilinear', align_corners=False)
        
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=(hx1.size(2), hx1.size(3)), mode='bilinear', align_corners=False)
        
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        
        return hx1d + hxin

# RSU-5 bloğu
class RSU5(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()
        
        self.rebnconvin = ConvBlock(in_ch, out_ch)
        
        self.rebnconv1 = ConvBlock(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv2 = ConvBlock(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv3 = ConvBlock(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv4 = ConvBlock(mid_ch, mid_ch)
        
        self.rebnconv5 = ConvBlock(mid_ch, mid_ch, dirate=2)
        
        self.rebnconv4d = ConvBlock(mid_ch*2, mid_ch)
        self.rebnconv3d = ConvBlock(mid_ch*2, mid_ch)
        self.rebnconv2d = ConvBlock(mid_ch*2, mid_ch)
        self.rebnconv1d = ConvBlock(mid_ch*2, out_ch)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        
        hx4 = self.rebnconv4(hx)
        
        hx5 = self.rebnconv5(hx4)
        
        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=(hx3.size(2), hx3.size(3)), mode='bilinear', align_corners=False)
        
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=(hx2.size(2), hx2.size(3)), mode='bilinear', align_corners=False)
        
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=(hx1.size(2), hx1.size(3)), mode='bilinear', align_corners=False)
        
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        
        return hx1d + hxin

# RSU-4 bloğu
class RSU4(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()
        
        self.rebnconvin = ConvBlock(in_ch, out_ch)
        
        self.rebnconv1 = ConvBlock(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv2 = ConvBlock(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv3 = ConvBlock(mid_ch, mid_ch)
        
        self.rebnconv4 = ConvBlock(mid_ch, mid_ch, dirate=2)
        
        self.rebnconv3d = ConvBlock(mid_ch*2, mid_ch)
        self.rebnconv2d = ConvBlock(mid_ch*2, mid_ch)
        self.rebnconv1d = ConvBlock(mid_ch*2, out_ch)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        
        hx3 = self.rebnconv3(hx)
        
        hx4 = self.rebnconv4(hx3)
        
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=(hx2.size(2), hx2.size(3)), mode='bilinear', align_corners=False)
        
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=(hx1.size(2), hx1.size(3)), mode='bilinear', align_corners=False)
        
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        
        return hx1d + hxin

# RSU-4F bloğu (dilation kullanır)
class RSU4F(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()
        
        self.rebnconvin = ConvBlock(in_ch, out_ch)
        
        self.rebnconv1 = ConvBlock(out_ch, mid_ch)
        self.rebnconv2 = ConvBlock(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = ConvBlock(mid_ch, mid_ch, dirate=4)
        self.rebnconv4 = ConvBlock(mid_ch, mid_ch, dirate=8)
        
        self.rebnconv3d = ConvBlock(mid_ch*2, mid_ch, dirate=4)
        self.rebnconv2d = ConvBlock(mid_ch*2, mid_ch, dirate=2)
        self.rebnconv1d = ConvBlock(mid_ch*2, out_ch)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)
        
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))
        
        return hx1d + hxin

# U²-Net Ana Model
class U2NET(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET, self).__init__()
        
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage6 = RSU4F(512, 256, 512)
        
        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)
        
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)
        
        self.outconv = nn.Conv2d(6*out_ch, out_ch, 1)
    
    def forward(self, x):
        # Float32'ye dönüştür
        x = x.float()
        
        hx = x
        
        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        
        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        
        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        
        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        
        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        
        # stage 6
        hx6 = self.stage6(hx)
        hx6up = F.interpolate(hx6, size=(hx5.size(2), hx5.size(3)), mode='bilinear', align_corners=False)
        
        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=(hx4.size(2), hx4.size(3)), mode='bilinear', align_corners=False)
        
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=(hx3.size(2), hx3.size(3)), mode='bilinear', align_corners=False)
        
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=(hx2.size(2), hx2.size(3)), mode='bilinear', align_corners=False)
        
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=(hx1.size(2), hx1.size(3)), mode='bilinear', align_corners=False)
        
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))
        
        # side output
        d1 = self.side1(hx1d)
        
        d2 = self.side2(hx2d)
        d2 = F.interpolate(d2, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        
        d3 = self.side3(hx3d)
        d3 = F.interpolate(d3, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        
        d4 = self.side4(hx4d)
        d4 = F.interpolate(d4, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        
        d5 = self.side5(hx5d)
        d5 = F.interpolate(d5, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        
        d6 = self.side6(hx6)
        d6 = F.interpolate(d6, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        
        # Tüm çıktıları float32'ye dönüştür
        d0 = torch.sigmoid(d0).float()
        d1 = torch.sigmoid(d1).float() 
        d2 = torch.sigmoid(d2).float()
        d3 = torch.sigmoid(d3).float()
        d4 = torch.sigmoid(d4).float()
        d5 = torch.sigmoid(d5).float()
        d6 = torch.sigmoid(d6).float()
        
        return d0, d1, d2, d3, d4, d5, d6

# Veri seti sınıfı
class LeafDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Dosya listelerini al ve sırala
        self.images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Dosya sayılarının eşit olduğunu kontrol et
        assert len(self.images) == len(self.masks), f"Görsel sayısı ({len(self.images)}) ve maske sayısı ({len(self.masks)}) eşit değil!"
        print(f"Toplam {len(self.images)} görsel-maske çifti bulundu.")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Görsel ve maskeyi yükle
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Maskeyi binary yap (0-1 arası) ve float32'ye dönüştür
        mask = (mask / 255.0).astype(np.float32)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
            # Mask'ın float32 olduğunu kesinleştir
            if mask.dtype != torch.float32:
                mask = mask.float()
        
        return image, mask.unsqueeze(0)  # Mask için kanal boyutu ekle

# Veri artırma (Data Augmentation)
def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

# Kayıp fonksiyonu (BCE + IoU)
class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.bce_loss = nn.BCELoss()
    
    def iou_loss(self, pred, target, smooth=1e-6):
        # Float32'ye dönüştür
        pred = pred.contiguous().float()
        target = target.contiguous().float()
        
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        return 1 - iou.mean()
    
    def forward(self, pred, target):
        # Float32'ye dönüştür
        pred = pred.float()
        target = target.float()
        
        bce = self.bce_loss(pred, target)
        iou = self.iou_loss(pred, target)
        return self.bce_weight * bce + (1 - self.bce_weight) * iou

# Metrik hesaplama
class DiceCoeff:
    def __init__(self, smooth=1e-6):
        self.smooth = smooth
    
    def __call__(self, pred, target):
        # Float32'ye dönüştür
        pred = pred.float()
        target = target.float()
        
        pred = (pred > 0.5).float()
        
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return dice.mean()

# IoU metriği
class IoUMetric:
    def __init__(self, smooth=1e-6):
        self.smooth = smooth
    
    def __call__(self, pred, target):
        # Float32'ye dönüştür
        pred = pred.float()
        target = target.float()
        
        pred = (pred > 0.5).float()
        
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        return iou.mean()

# Eğitim fonksiyonu
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    dice_metric = DiceCoeff()
    iou_metric = IoUMetric()
    total_dice = 0
    total_iou = 0
    
    pbar = tqdm(dataloader, desc="Eğitim")
    for images, masks in pbar:
        # Float32'ye dönüştür
        images = images.float().to(device)
        masks = masks.float().to(device)
        
        optimizer.zero_grad()
        
        # Model çıktıları (7 farklı çıktı)
        d0, d1, d2, d3, d4, d5, d6 = model(images)
        
        # Çok çıktılı kayıp hesaplama
        loss0 = criterion(d0, masks)
        loss1 = criterion(d1, masks)
        loss2 = criterion(d2, masks)
        loss3 = criterion(d3, masks)
        loss4 = criterion(d4, masks)
        loss5 = criterion(d5, masks)
        loss6 = criterion(d6, masks)
        
        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Metrik hesaplama (ana çıktı d0 için)
        dice_score = dice_metric(d0, masks)
        iou_score = iou_metric(d0, masks)
        total_dice += dice_score.item()
        total_iou += iou_score.item()
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Dice': f'{dice_score.item():.4f}',
            'IoU': f'{iou_score.item():.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    
    return avg_loss, avg_dice, avg_iou

# Doğrulama fonksiyonu
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    dice_metric = DiceCoeff()
    iou_metric = IoUMetric()
    total_dice = 0
    total_iou = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Doğrulama")
        for images, masks in pbar:
            # Float32'ye dönüştür
            images = images.float().to(device)
            masks = masks.float().to(device)
            
            # Model çıktıları
            d0, d1, d2, d3, d4, d5, d6 = model(images)
            
            # Kayıp hesaplama
            loss0 = criterion(d0, masks)
            loss1 = criterion(d1, masks)
            loss2 = criterion(d2, masks)
            loss3 = criterion(d3, masks)
            loss4 = criterion(d4, masks)
            loss5 = criterion(d5, masks)
            loss6 = criterion(d6, masks)
            
            loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            
            total_loss += loss.item()
            
            # Metrik hesaplama
            dice_score = dice_metric(d0, masks)
            iou_score = iou_metric(d0, masks)
            total_dice += dice_score.item()
            total_iou += iou_score.item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{dice_score.item():.4f}',
                'IoU': f'{iou_score.item():.4f}'
            })
    
    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    
    return avg_loss, avg_dice, avg_iou

# Sonuçları görselleştirme
def visualize_predictions(model, dataloader, device, num_samples=4):
    model.eval()
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            images, masks = images.to(device), masks.to(device)
            
            # Tek bir örnek al
            image = images[0:1]
            mask = masks[0:1]
            
            # Tahmin
            pred, _, _, _, _, _, _ = model(image)
            
            # Tensörları numpy'ye çevir
            image_np = image[0].cpu().permute(1, 2, 0).numpy()
            mask_np = mask[0, 0].cpu().numpy()
            pred_np = pred[0, 0].cpu().numpy()
            
            # Normalizasyonu geri al
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = std * image_np + mean
            image_np = np.clip(image_np, 0, 1)
            
            # Görselleri çiz
            if num_samples == 1:
                axes = [axes]
            
            axes[i][0].imshow(image_np)
            axes[i][0].set_title('Orijinal Görsel')
            axes[i][0].axis('off')
            
            axes[i][1].imshow(mask_np, cmap='gray')
            axes[i][1].set_title('Gerçek Maske')
            axes[i][1].axis('off')
            
            axes[i][2].imshow(pred_np, cmap='gray')
            axes[i][2].set_title('Tahmin Edilen Maske')
            axes[i][2].axis('off')
            
            # Tahmin edilen maskeyi binary yap
            pred_binary = (pred_np > 0.5).astype(np.uint8)
            axes[i][3].imshow(pred_binary, cmap='gray')
            axes[i][3].set_title('Binary Tahmin')
            axes[i][3].axis('off')
    
    plt.tight_layout()
    plt.show()

# Model kaydetme
def save_model(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Model kaydedildi: {path}")

# Model yükleme
def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

# Ana eğitim fonksiyonu
def main():
    print("YAPRAK SEGMENTASYONU UYGULAMASI")
    
    # Veri setindeki görselleri al
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not image_files:
        print(f"HATA: {IMAGE_DIR} dizininde görsel bulunamadı!")
        exit()
    
    print(f"Veri setinde toplam {len(image_files)} görsel bulundu.")
    
    # Segmentasyon yöntemini seç
    print("\nSegmentasyon yöntemini seçiniz:")
    print("1) Makine Öğrenmesi Modeli (U²-Net)")
    print("2) Klasik Görüntü İşleme (HSV Renk Filtresi)")
    method_choice = input("Seçiminiz (1 veya 2): ")
    
    if method_choice == "1":
        # Eğitilmiş modeli yükle
        print("\nMakine öğrenmesi modeli yükleniyor...")
        model = U2NET(in_ch=3, out_ch=1).to(device)
        checkpoint = torch.load('best_u2net_leaf_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model başarıyla yüklendi!")
    
    # Test edilecek görsellerin aralığını kullanıcıdan al
    try:
        print("\nHangi görselleri işlemek istiyorsunuz?")
        print("1) İlk 10 görseli işle")
        print("2) Özel bir aralık belirle")
        choice = input("Seçiminiz (1 veya 2): ")
        
        if choice == "1":
            start_idx = 0
            end_idx = 10
        elif choice == "2":
            start_idx = int(input(f"Başlangıç indeksi (0-{len(image_files)-1}): "))
            end_idx = int(input(f"Bitiş indeksi (dahil) ({start_idx+1}-{len(image_files)-1}): ")) + 1  # Dahil olması için +1
        else:
            print("Geçersiz seçim. İlk 10 görsel işlenecek.")
            start_idx = 0
            end_idx = 10
    except:
        print("Geçersiz giriş. İlk 10 görsel işlenecek.")
        start_idx = 0
        end_idx = 10
    
    # Görsel sayısına göre kontrol et
    if start_idx < 0:
        start_idx = 0
    if end_idx > len(image_files):
        end_idx = len(image_files)
    
    print(f"\nİşlenecek görseller: {start_idx} ile {end_idx-1} indeks aralığı ({end_idx-start_idx} görsel)")
    
    # Belirtilen aralıktaki görselleri işle
    for idx in range(start_idx, end_idx):
        test_image_path = os.path.join(IMAGE_DIR, image_files[idx])
        output_path = f'segmente_yaprak_{idx}.jpg'
        
        print(f"\nGörsel {idx}: {image_files[idx]} işleniyor...")
        
        # Seçilen yönteme göre segmentasyon yap
        if method_choice == "1":
            # Makine öğrenmesi modeli ile segmentasyon
            predict_single_image(model, test_image_path, device, output_path)
        else:
            # Klasik görüntü işleme ile segmentasyon
            segment_leaf_classical(test_image_path, output_path)
        
        print(f"Sonuç kaydedildi: {output_path}")
    
    print("\nTüm görseller işlendi! Sonuçlar kaydedildi.")

# Test fonksiyonu (yeni görsel için tahmin)
def predict_single_image(model, image_path, device, save_path=None):
    """Tek bir görsel için segmentasyon tahmini
    Adaptif yaklaşım - farklı eşik değerleri deneyerek en iyi sonucu seçer
    """
    model.eval()
    
    # Görseli yükle
    image = cv2.imread(image_path)
    original_image = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Transform uygula
    transform = get_transforms(False)
    augmented = transform(image=image_rgb)
    image_tensor = augmented['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Tahmin
        pred, _, _, _, _, _, _ = model(image_tensor)
        pred_np = pred[0, 0].cpu().numpy()
    
    h, w = image.shape[:2]
    
    # Farklı eşik değerleri ile deneme yap
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    best_mask = None
    best_area = 0
    best_threshold = 0.5
    
    masks_to_show = []
    
    for threshold in thresholds:
        # Bu eşik ile maske oluştur
        initial_mask = (pred_np > threshold).astype(np.uint8)
        initial_mask_resized = cv2.resize(initial_mask, (w, h))
        
        # Morfolojik işlemler
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_medium = np.ones((7, 7), np.uint8)
        
        # Delikleri kapat
        closed_mask = cv2.morphologyEx(initial_mask_resized, cv2.MORPH_CLOSE, kernel_small, iterations=2)
        closed_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
        
        # Hafif pürüzsüzleştirme
        blurred_mask = cv2.GaussianBlur(closed_mask.astype(np.float32), (5, 5), 0)
        smooth_mask = (blurred_mask > 0.5).astype(np.uint8)
        
        # Kontur işlemleri
        contours, _ = cv2.findContours(smooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Kontur kalitesini değerlendir
            epsilon = 0.001 * cv2.arcLength(largest_contour, True)
            approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            refined_mask = np.zeros_like(smooth_mask)
            cv2.drawContours(refined_mask, [approx_contour], 0, 1, -1)
            
            # En büyük ve makul alan boyutuna sahip maskeyi seç
            total_pixels = h * w
            area_ratio = area / total_pixels
            
            # Alan oranı %5 ile %80 arasında olmalı (çok küçük veya çok büyük olmasın)
            if 0.05 < area_ratio < 0.8 and area > best_area:
                best_area = area
                best_mask = refined_mask
                best_threshold = threshold
        
        # İlk 3 maskeyi görselleştirme için sakla
        if len(masks_to_show) < 3:
            masks_to_show.append((threshold, smooth_mask))
    
    # Eğer hiç uygun maske bulunamadıysa, orta eşik değerini kullan
    if best_mask is None:
        threshold = 0.5
        initial_mask = (pred_np > threshold).astype(np.uint8)
        initial_mask_resized = cv2.resize(initial_mask, (w, h))
        
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_medium = np.ones((7, 7), np.uint8)
        
        closed_mask = cv2.morphologyEx(initial_mask_resized, cv2.MORPH_CLOSE, kernel_small, iterations=2)
        closed_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
        
        blurred_mask = cv2.GaussianBlur(closed_mask.astype(np.float32), (5, 5), 0)
        best_mask = (blurred_mask > 0.5).astype(np.uint8)
        best_threshold = threshold
    
    # Sonucu oluştur
    mask_3d = np.stack([best_mask] * 3, axis=2)
    result = original_image * mask_3d
    
    # Sonuçları göster
    plt.figure(figsize=(20, 8))
    
    # Üst satır: Farklı eşik değerleri ile denemeler
    plt.subplot(2, 5, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Orijinal Görsel')
    plt.axis('off')
    
    for i, (thresh, mask) in enumerate(masks_to_show):
        plt.subplot(2, 5, i+2)
        plt.imshow(mask, cmap='gray')
        plt.title(f'Eşik: {thresh}')
        plt.axis('off')
    
    # Alt satır: En iyi sonuç
    plt.subplot(2, 5, 6)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Orijinal')
    plt.axis('off')
    
    plt.subplot(2, 5, 7)
    plt.imshow(best_mask, cmap='gray')
    plt.title(f'En İyi Maske\n(Eşik: {best_threshold})')
    plt.axis('off')
    
    plt.subplot(2, 5, 8)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Sonuç')
    plt.axis('off')
    
    # Alan bilgisi
    area_ratio = (best_area / (h * w)) * 100
    plt.subplot(2, 5, 9)
    plt.text(0.1, 0.7, f'Seçilen Eşik: {best_threshold}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.5, f'Yaprak Alanı: %{area_ratio:.1f}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.3, f'Piksel Sayısı: {int(best_area)}', fontsize=12, transform=plt.gca().transAxes)
    plt.title('İstatistikler')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"En iyi sonuç için eşik değeri: {best_threshold}")
    print(f"Yaprak alanı: %{area_ratio:.1f}")
    
    # Kaydedilmek istenirse
    if save_path:
        cv2.imwrite(save_path, result)
        print(f"Sonuç kaydedildi: {save_path}")
    
    return result

# Alternatif yaklaşım: Klasik görüntü işleme ile yaprak segmentasyonu
def segment_leaf_classical(image_path, save_path=None):
    """Klasik görüntü işleme teknikleri kullanarak yaprak segmentasyonu
    Bu fonksiyon makine öğrenmesi modeli kullanmaz, doğrudan görüntü üzerinde işlem yapar.
    """
    # Görseli oku
    image = cv2.imread(image_path)
    if image is None:
        print(f"HATA: Görsel yüklenemedi: {image_path}")
        return None
    
    # RGB ve HSV formatlarına dönüştür
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Yeşil ve kahverengi yaprakları tespit etmek için HSV eşikleme
    # Yeşil yapraklar için HSV eşikleme
    lower_green = np.array([25, 20, 20])
    upper_green = np.array([100, 255, 255])
    mask_green = cv2.inRange(image_hsv, lower_green, upper_green)
    
    # Kahverengi/sarı yapraklar için HSV eşikleme
    lower_brown = np.array([10, 20, 20])
    upper_brown = np.array([30, 255, 255])
    mask_brown = cv2.inRange(image_hsv, lower_brown, upper_brown)
    
    # Maskeleri birleştir
    mask_combined = cv2.bitwise_or(mask_green, mask_brown)
    
    # Gürültü temizleme
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Dış hatları bul
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # En büyük konturu bul (en büyük yaprak)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Konturun içini doldur
        final_mask = np.zeros_like(mask_cleaned)
        cv2.drawContours(final_mask, [largest_contour], 0, 255, -1)
        
        # Maske ile orijinal görüntüyü birleştir
        mask_3d = np.stack([final_mask/255] * 3, axis=2).astype(np.uint8)
        result = image * mask_3d
        
        # Sonuçları göster
        plt.figure(figsize=(16, 4))
        plt.subplot(1, 4, 1)
        plt.imshow(image_rgb)
        plt.title('Orijinal Görsel')
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(mask_combined, cmap='gray')
        plt.title('Birleşik Maske\n(Yeşil + Kahverengi)')
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.imshow(final_mask, cmap='gray')
        plt.title('Düzeltilmiş Maske')
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title('Sonuç: Yaprak Orijinal\nArka Plan Siyah')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Kaydedilmek istenirse
        if save_path:
            cv2.imwrite(save_path, result)
            print(f"Sonuç kaydedildi: {save_path}")
        
        return result
    else:
        print("UYARI: Yaprak tespit edilemedi!")
        return None

# Kullanım örneği
if __name__ == "__main__":
    main()