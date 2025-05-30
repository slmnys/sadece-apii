from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import tensorflow as tf
from PIL import Image
import os
from flask_cors import CORS
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import requests

def download_file_from_google_drive(url, destination):
    if not os.path.exists(destination):
        print(f"{destination} indiriliyor...")
        response = requests.get(url, stream=True)
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"{destination} indirildi.")

# Model dosyası yoksa indir
download_file_from_google_drive(
    "https://drive.google.com/uc?export=download&id=143otNbtJIUk255fviGREVu970tJlva2C",
    "best_u2net_leaf_model.pth"
)
download_file_from_google_drive(
    "https://drive.google.com/uc?export=download&id=1C_aykNRjBR0KTyQz6uzVYd1MTQEauerv",
    "leaf_model.tflite"
)

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# GPU kontrolü
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Kullanılan cihaz: {device}")

# U²-Net Model Sınıfları (local_yaprak_segmentasyon.py'den)
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

# U²-Net Segmentasyon Modeli Yükleme
U2NET_MODEL_PATH = 'best_u2net_leaf_model.pth'
if not os.path.exists(U2NET_MODEL_PATH):
    raise FileNotFoundError(f"{U2NET_MODEL_PATH} bulunamadı!")

print("U²-Net modeli yükleniyor...")
u2net_model = U2NET(in_ch=3, out_ch=1).to(device)
checkpoint = torch.load(U2NET_MODEL_PATH, map_location=device)
u2net_model.load_state_dict(checkpoint['model_state_dict'])
u2net_model.eval()
print("✅ U²-Net modeli başarıyla yüklendi!")

# TensorFlow Lite Sınıflandırma Modeli
TFLITE_MODEL_PATH = 'leaf_model.tflite'
if not os.path.exists(TFLITE_MODEL_PATH):
    raise FileNotFoundError(f"{TFLITE_MODEL_PATH} bulunamadı!")

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASS_NAMES = ['apple', 'grape', 'orange', 'soybean', 'tomato']

def preprocess_for_u2net(image, size=320):
    """U²-Net için görsel ön işleme"""
    # Boyutlandır
    image_resized = cv2.resize(image, (size, size))
    
    # Normalize et
    image_normalized = image_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_normalized = (image_normalized - mean) / std
    
    # Tensor'e çevir
    image_tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1)).unsqueeze(0)
    
    return image_tensor

def u2net_leaf_segmentation(image):
    """U²-Net ile yaprak segmentasyonu - Adaptif eşik ile"""
    try:
        # Görseli hazırla
        image_tensor = preprocess_for_u2net(image)
        image_tensor = image_tensor.to(device)
        
        # Model tahmini
        with torch.no_grad():
            pred, _, _, _, _, _, _ = u2net_model(image_tensor)
            pred_np = pred[0, 0].cpu().numpy()
        
        h, w = image.shape[:2]
        
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
        result = image * mask_3d
        
        print(f"[U2NET] Segmentasyon tamamlandı. Eşik: {best_threshold}, Alan: %{(best_area/(h*w))*100:.1f}")
        
        return result, best_mask
        
    except Exception as e:
        print(f"[U2NET] Segmentasyon hatası: {e}")
        # Hata durumunda orijinal görseli döndür
        return image, np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255

def preprocess_for_model(image, target_size=(224, 224)):
    """TensorFlow Lite modeli için ön işleme"""
    image_pil = Image.fromarray(image)
    try:
        image_resized = image_pil.resize(target_size, Image.Resampling.LANCZOS)
    except AttributeError:
        image_resized = image_pil.resize(target_size, Image.LANCZOS)
    image_array = np.array(image_resized, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)

def predict_leaf_type(segmented_image):
    """Segmente edilmiş yaprak görselini sınıflandır"""
    try:
        gray = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)
        if cv2.countNonZero(gray) < 1000:
            print("[MODEL] Yeterli yaprak bulunamadı")
            return {"status": "error", "message": "Görselde yeterli yaprak bulunamadı"}
        
        input_data = preprocess_for_model(segmented_image)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        class_probabilities = {
            CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))
        }
        
        predicted_index = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(predictions[predicted_index])
        
        print(f"[MODEL] Tahmin: {predicted_class}, Güven: {confidence:.2f}")
        
        # Güven %55'in altındaysa belirsiz olarak işaretle
        if confidence < 0.55:
            return {
                "status": "success",
                "prediction": {
                    "predicted_class": "belirsiz",
                    "confidence": confidence,
                    "confidence_percentage": confidence * 100,
                    "all_probabilities": class_probabilities,
                    "message": "Bu görsel büyük ihtimalle belirsiz bir cisimdir."
                }
            }
        
        return {
            "status": "success",
            "prediction": {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "confidence_percentage": confidence * 100,
                "all_probabilities": class_probabilities
            }
        }
    except Exception as e:
        print(f"[MODEL] Hata: {e}")
        return {"status": "error", "message": f"Model tahmini başarısız: {e}"}

@app.route('/classify', methods=['POST'])
def classify_leaf():
    try:
        print("\n[SERVER] İstek alındı")
        
        if 'image' not in request.files:
            print("[HATA] 'image' anahtarı yok")
            return jsonify({"error": "Görsel yüklenmedi"}), 400

        file = request.files['image']
        print("[INFO] Dosya alındı:", file.filename)

        if file.filename == '':
            print("[HATA] Dosya adı boş")
            return jsonify({"error": "Dosya adı boş"}), 400

        image_bytes = file.read()
        print("[INFO] Dosya boyutu:", len(image_bytes))

        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            print("[HATA] cv2.imdecode başarısız")
            return jsonify({"error": "Görsel okunamadı, desteklenmeyen format"}), 400

        print("[INFO] cv2.imdecode başarılı")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # U²-Net ile segmentasyon
        print("[INFO] U²-Net segmentasyon başlıyor...")
        segmented_image, mask = u2net_leaf_segmentation(image)
        print("[INFO] U²-Net segmentasyon tamamlandı")

        # Sınıflandırma
        prediction_result = predict_leaf_type(segmented_image)

        if prediction_result["status"] != "success":
            print("[HATA] Model sonucu başarısız:", prediction_result["message"])
            return jsonify(prediction_result), 400

        print("[INFO] Tahmin başarılı")

        # Base64 kodlama (Flutter için)
        _, segmented_buf = cv2.imencode('.jpg', cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        segmented_base64 = base64.b64encode(segmented_buf).decode('utf-8')
        
        _, mask_buf = cv2.imencode('.jpg', mask, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        mask_base64 = base64.b64encode(mask_buf).decode('utf-8')

        response_data = {
            "status": "success",
            "prediction": prediction_result["prediction"],
            "segmented_image": segmented_base64,
            "mask": mask_base64
        }
        
        print(f"[INFO] Response boyutu: {len(str(response_data))} karakter")
        return jsonify(response_data)

    except Exception as e:
        print(f"[SERVER] İşlem hatası: {e}")
        return jsonify({"error": f"Hata oluştu: {e}"}), 500

if __name__ == '__main__':
    print(f"U²-Net Segmentasyon Modeli: {U2NET_MODEL_PATH}")
    print(f"TensorFlow Lite Sınıflandırma Modeli: {TFLITE_MODEL_PATH}")
    print(f"Desteklenen sınıflar: {CLASS_NAMES}")
    print("Server başlatılıyor...")
    print("Emülatör için: http://10.0.2.2:5000")
    print("Gerçek cihaz için: http://192.168.1.22:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)