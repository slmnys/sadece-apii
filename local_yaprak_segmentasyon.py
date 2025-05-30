import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# GPU kontrolü
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Kullanılan cihaz: {device}")

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

# RSU-7 bloğu (sadece gerekli kısımlar)
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

# Diğer RSU blokları (kısaltılmış)
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
        x = x.float()
        hx = x
        
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        
        hx6 = self.stage6(hx)
        hx6up = F.interpolate(hx6, size=(hx5.size(2), hx5.size(3)), mode='bilinear', align_corners=False)
        
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=(hx4.size(2), hx4.size(3)), mode='bilinear', align_corners=False)
        
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=(hx3.size(2), hx3.size(3)), mode='bilinear', align_corners=False)
        
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=(hx2.size(2), hx2.size(3)), mode='bilinear', align_corners=False)
        
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=(hx1.size(2), hx1.size(3)), mode='bilinear', align_corners=False)
        
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))
        
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
        
        d0 = torch.sigmoid(d0).float()
        d1 = torch.sigmoid(d1).float() 
        d2 = torch.sigmoid(d2).float()
        d3 = torch.sigmoid(d3).float()
        d4 = torch.sigmoid(d4).float()
        d5 = torch.sigmoid(d5).float()
        d6 = torch.sigmoid(d6).float()
        
        return d0, d1, d2, d3, d4, d5, d6

def preprocess_image(image_path, size=320):
    """Görseli model için hazırla - Türkçe karakter desteği ile"""
    # Türkçe karakter sorunu için numpy ile oku
    try:
        # OpenCV yerine numpy ile oku
        with open(image_path, 'rb') as f:
            image_data = np.frombuffer(f.read(), np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError(f"Görsel okunamadı: {image_path}")
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Görsel okuma hatası: {e}")
        return None, None
    
    # Boyutlandır
    image_resized = cv2.resize(image_rgb, (size, size))
    
    # Normalize et
    image_normalized = image_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_normalized = (image_normalized - mean) / std
    
    # Tensor'e çevir
    image_tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1)).unsqueeze(0)
    
    return image_tensor, image_rgb

def segment_leaf(model, image_path, output_path=None):
    """Yaprak segmentasyonu yap"""
    
    # Görseli hazırla
    image_tensor, original_image = preprocess_image(image_path)
    
    # Hata kontrolü
    if image_tensor is None or original_image is None:
        print("❌ Görsel işlenemedi!")
        return None
        
    image_tensor = image_tensor.to(device)
    
    # Model tahmini
    with torch.no_grad():
        pred, _, _, _, _, _, _ = model(image_tensor)
        pred_np = pred[0, 0].cpu().numpy()
    
    h, w = original_image.shape[:2]
    
    # Farklı eşik değerleri dene
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    best_mask = None
    best_area = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        # Maske oluştur
        initial_mask = (pred_np > threshold).astype(np.uint8)
        initial_mask_resized = cv2.resize(initial_mask, (w, h))
        
        # Morfolojik işlemler
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_medium = np.ones((7, 7), np.uint8)
        
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
            
            epsilon = 0.001 * cv2.arcLength(largest_contour, True)
            approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            refined_mask = np.zeros_like(smooth_mask)
            cv2.drawContours(refined_mask, [approx_contour], 0, 1, -1)
            
            # En iyi maskeyi seç
            total_pixels = h * w
            area_ratio = area / total_pixels
            
            if 0.05 < area_ratio < 0.8 and area > best_area:
                best_area = area
                best_mask = refined_mask
                best_threshold = threshold
    
    # Eğer hiç uygun maske bulunamadıysa
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
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('Orijinal Görsel')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(best_mask, cmap='gray')
    plt.title(f'Maske (Eşik: {best_threshold})')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(result)
    plt.title('Segmente Edilmiş Yaprak')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Kaydet
    if output_path:
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_bgr)
        print(f"Sonuç kaydedildi: {output_path}")
    
    area_ratio = (best_area / (h * w)) * 100
    print(f"Kullanılan eşik: {best_threshold}")
    print(f"Yaprak alanı: %{area_ratio:.1f}")
    
    return result

# Ana kullanım fonksiyonu
def main():
    print("🍃 Yaprak Segmentasyon Uygulaması")
    print("=" * 40)
    
    # Model dosyası kontrolü
    model_path = "best_u2net_leaf_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ HATA: Model dosyası bulunamadı: {model_path}")
        print("Lütfen 'best_u2net_leaf_model.pth' dosyasını bu klasöre koyun.")
        return
    
    # Modeli yükle
    print("Model yükleniyor...")
    model = U2NET(in_ch=3, out_ch=1).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Model başarıyla yüklendi!")
    
    # Görsel yolu al
    image_path = input("Yaprak görselinin yolunu girin: ")
    
    # Dosya yolu kontrolü
    if not os.path.exists(image_path):
        print(f"❌ HATA: Görsel bulunamadı: {image_path}")
        print("💡 İpucu: Dosya yolunda Türkçe karakter varsa tırnak içinde yazın")
        print("💡 Örnek: \"C:\\Users\\kullanici\\Desktop\\yaprak.jpg\"")
        return
    
    # Dosya uzantısı kontrolü
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    file_ext = os.path.splitext(image_path)[1].lower()
    if file_ext not in valid_extensions:
        print(f"❌ HATA: Desteklenmeyen dosya formatı: {file_ext}")
        print(f"✅ Desteklenen formatlar: {', '.join(valid_extensions)}")
        return
    
    # Çıktı yolu al
    output_path = input("Sonucun kaydedileceği dosya adı (örn: sonuc.jpg): ")
    if not output_path:
        output_path = "segmented_leaf.jpg"
    
    # Segmentasyon yap
    print("Segmentasyon yapılıyor...")
    result = segment_leaf(model, image_path, output_path)
    
    print("🎉 İşlem tamamlandı!")

if __name__ == "__main__":
    main() 
    