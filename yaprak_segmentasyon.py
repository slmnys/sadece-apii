import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# --- Parametreler ---
IMG_SIZE = 128  # Daha küçük boyut - daha hızlı
BATCH_SIZE = 16  # Daha büyük batch
EPOCHS = 50

# --- Dataset Yolu ---
dataset_path = "/kaggle/input/dataset/dataset"
image_dir = os.path.join(dataset_path, "images")
mask_dir = os.path.join(dataset_path, "mask")

print(f"Görsel klasörü: {image_dir}")
print(f"Maske klasörü: {mask_dir}")

# --- Focal Loss (Zor örneklere odaklanır) ---
def focal_loss(alpha=0.25, gamma=2.0):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Calculate focal loss
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_loss_val = -alpha_t * tf.pow((1 - p_t), gamma) * tf.log(p_t)
        
        return tf.reduce_mean(focal_loss_val)
    return focal_loss_fixed

# --- Dice + BCE Kombinasyonu ---
def dice_bce_loss(y_true, y_pred):
    # Dice loss
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true, axis=[1,2,3]) + tf.reduce_sum(y_pred, axis=[1,2,3])
    dice = tf.reduce_mean((2. * intersection + 1) / (union + 1))
    dice_loss = 1 - dice
    
    # Binary crossentropy
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)
    
    # Kombinasyon
    return 0.5 * dice_loss + 0.5 * bce

# --- IoU Metriği ---
def iou_metric(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true, axis=[1,2,3]) + tf.reduce_sum(y_pred, axis=[1,2,3]) - intersection
    iou = tf.reduce_mean((intersection + 1) / (union + 1))
    return iou

# --- Gelişmiş Preprocessing ---
def preprocess_image(image):
    # Kontrastı artır
    image = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
    
    # Gürültüyü azalt
    image = cv2.bilateralFilter(image, 9, 75, 75)
    
    return image

def preprocess_mask(mask):
    # Morfolojik işlemler
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask

# --- Basit ama Etkili Veri Yükleme ---
def load_data_simple():
    images = []
    masks = []
    
    try:
        image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        print(f"Bulunan görsel sayısı: {len(image_files)}")
        
        # Sadece ilk 500 veriyi al (hızlı test için)
        num_samples = min(500, len(image_files))
        
        for i in range(num_samples):
            img_file = image_files[i]
            mask_file = mask_files[i]
            
            try:
                # Görseli yükle ve işle
                img_path = os.path.join(image_dir, img_file)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = preprocess_image(img)  # Gelişmiş preprocessing
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img.astype(np.float32) / 255.0
                
                # Maskeyi yükle ve işle
                mask_path = os.path.join(mask_dir, mask_file)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = preprocess_mask(mask)  # Gelişmiş preprocessing
                mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
                mask = (mask > 127).astype(np.float32)
                mask = np.expand_dims(mask, axis=-1)
                
                images.append(img)
                masks.append(mask)
                
                if (i + 1) % 100 == 0:
                    print(f"📊 Yüklendi: {i+1}/{num_samples}")
                    
            except Exception as e:
                print(f"❌ Hata {img_file}: {e}")
                continue
        
        print(f"✅ Toplam yüklenen: {len(images)}")
        
        # Numpy array'e çevir
        images = np.array(images, dtype=np.float32)
        masks = np.array(masks, dtype=np.float32)
        
        return images, masks
        
    except Exception as e:
        print(f"❌ Genel hata: {e}")
        return np.array([]), np.array([])

# --- Basit ama Etkili U-Net ---
def create_simple_unet():
    inputs = Input((IMG_SIZE, IMG_SIZE, 3))
    
    # Encoder
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Bottleneck
    conv4 = Conv2D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, 3, activation='relu', padding='same')(conv4)
    conv4 = Dropout(0.5)(conv4)
    
    # Decoder
    up5 = UpSampling2D(size=(2, 2))(conv4)
    up5 = concatenate([up5, conv3], axis=-1)
    conv5 = Conv2D(128, 3, activation='relu', padding='same')(up5)
    conv5 = Conv2D(128, 3, activation='relu', padding='same')(conv5)
    
    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = concatenate([up6, conv2], axis=-1)
    conv6 = Conv2D(64, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(64, 3, activation='relu', padding='same')(conv6)
    
    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = concatenate([up7, conv1], axis=-1)
    conv7 = Conv2D(32, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(32, 3, activation='relu', padding='same')(conv7)
    
    # Output
    outputs = Conv2D(1, 1, activation='sigmoid')(conv7)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# --- Sonuçları Göster ---
def show_results_improved(X, Y, model, num_samples=5):
    indices = np.random.choice(len(X), num_samples, replace=False)
    preds = model.predict(X[indices])
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    for i, idx in enumerate(indices):
        # Orijinal görsel
        axes[i, 0].imshow(X[idx])
        axes[i, 0].set_title('Orijinal Görsel')
        axes[i, 0].axis('off')
        
        # Gerçek maske
        axes[i, 1].imshow(Y[idx].squeeze(), cmap='gray')
        axes[i, 1].set_title('Gerçek Maske')
        axes[i, 1].axis('off')
        
        # Tahmin edilen maske - daha iyi threshold
        pred_mask = (preds[i].squeeze() > 0.3).astype(np.float32)  # 0.5 yerine 0.3
        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title('Tahmin Edilen Maske')
        axes[i, 2].axis('off')
        
        # Segmente edilmiş yaprak
        segmented = X[idx].copy()
        mask_3d = np.stack([pred_mask]*3, axis=-1)
        segmented = segmented * mask_3d
        axes[i, 3].imshow(segmented)
        axes[i, 3].set_title('Segmente Edilmiş Yaprak\n(Siyah Arka Plan)')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.show()

# --- ANA PROGRAM ---
print("🚀 Basit ama etkili model yükleniyor...")
X, Y = load_data_simple()

if len(X) == 0:
    print("❌ Veri yüklenemedi!")
else:
    print(f"✅ Veri yüklendi: {X.shape}, {Y.shape}")
    
    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
    
    print(f"Eğitim: {X_train.shape}, Test: {X_test.shape}")
    
    # Model oluştur
    print("🤖 Basit U-Net modeli oluşturuluyor...")
    model = create_simple_unet()
    
    # Compile - Dice+BCE Loss ile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=dice_bce_loss,
        metrics=[iou_metric, 'accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6, monitor='val_loss')
    ]
    
    # Eğitim
    print("🎯 Model eğitiliyor...")
    history = model.fit(
        X_train, Y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, Y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Grafikler
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Eğitim Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('Dice+BCE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['iou_metric'], label='Eğitim IoU')
    plt.plot(history.history['val_iou_metric'], label='Test IoU')
    plt.title('IoU Metric')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['accuracy'], label='Eğitim Accuracy')
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Test sonuçları
    print("\n📊 Final test sonuçları:")
    test_results = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Test Loss: {test_results[0]:.4f}")
    print(f"Test IoU: {test_results[1]:.4f}")
    print(f"Test Accuracy: {test_results[2]:.4f}")
    
    # Görselleştirme
    print("\n🖼️ İyileştirilmiş segmentasyon sonuçları:")
    show_results_improved(X_test, Y_test, model, num_samples=5)
    
    # Model kaydet
    model.save('basit_etkili_yaprak_model.h5')
    print("💾 Model kaydedildi!")
    
print("🎉 Basit ama etkili eğitim tamamlandı!") 