from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import tensorflow as tf
from PIL import Image
import os
from flask_cors import CORS
import time

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Model yükleme
MODEL_PATH = 'leaf_model.tflite'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} bulunamadı!")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASS_NAMES = ['apple', 'grape', 'orange', 'soybean', 'tomato']

def improved_leaf_segmentation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_green1 = np.array([25, 40, 40])
    upper_green1 = np.array([85, 255, 255])
    mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
    lower_green2 = np.array([15, 30, 30])
    upper_green2 = np.array([35, 255, 255])
    mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.medianBlur(mask, 5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        refined_mask = np.zeros_like(mask)
        min_area = image.shape[0] * image.shape[1] * 0.01
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.drawContours(refined_mask, [contour], -1, 255, -1)
        if cv2.countNonZero(refined_mask) > 0:
            mask = refined_mask
    result = cv2.bitwise_and(image, image, mask=mask)
    background = np.zeros_like(image)
    segmented_leaf = np.where(mask[..., None] == 0, background, result)
    return segmented_leaf, mask

def preprocess_for_model(image, target_size=(224, 224)):
    image_pil = Image.fromarray(image)
    try:
        image_resized = image_pil.resize(target_size, Image.Resampling.LANCZOS)
    except AttributeError:
        # Pillow < 10.0.0 için uyumlu yapı
        image_resized = image_pil.resize(target_size, Image.LANCZOS)
    image_array = np.array(image_resized, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def predict_leaf_type(segmented_image):
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
        # Güven %60'ın altındaysa belirsiz olarak işaretle
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
        print("Headers:", request.headers)
        print("Files:", request.files)
        print("Form:", request.form)

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
        segmented_image, mask = improved_leaf_segmentation(image)

        print("[INFO] Segmentasyon tamamlandı")
        prediction_result = predict_leaf_type(segmented_image)

        if prediction_result["status"] != "success":
            print("[HATA] Model sonucu başarısız:", prediction_result["message"])
            return jsonify(prediction_result), 400

        print("[INFO] Tahmin başarılı")

        # Emülatör için daha düşük kalite (daha küçük boyut)
        _, segmented_buf = cv2.imencode('.jpg', cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        segmented_base64 = base64.b64encode(segmented_buf).decode('utf-8')
        _, mask_buf = cv2.imencode('.jpg', mask, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
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
    print(f"Model yüklendi: {MODEL_PATH}")
    print(f"Desteklenen sınıflar: {CLASS_NAMES}")
    print("Server başlatılıyor...")
    print("Emülatör için: http://10.0.2.2:5000")
    print("Gerçek cihaz için: http://192.168.1.22:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)