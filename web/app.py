import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import os
import torch
import cv2
import numpy as np
from torchvision.models import efficientnet_b2
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
import albumentations as A
from PIL import Image
import io
from werkzeug.utils import secure_filename
import requests
import base64
import json
import datetime
import openai

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# ESP32-CAM IP address
ESP32_IP = "192.168.0.106"

# Tạo thư mục uploads nếu chưa tồn tại
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Thông tin bệnh
DISEASE_INFO = {
    "BrownSpot": {
        "desc": "Bệnh đốm nâu gây ra các vết nâu trên lá, làm giảm năng suất.",
        "solution": "Sử dụng thuốc trừ nấm, bón phân cân đối, giữ ruộng thông thoáng.",
        "color": (0, 0, 255)  # Màu đỏ cho đốm nâu
    },
    "Healthy": {
        "desc": "Cây lúa khỏe mạnh, không có dấu hiệu bệnh.",
        "solution": "Tiếp tục chăm sóc bình thường.",
        "color": (0, 255, 0)  # Màu xanh lá cho cây khỏe
    },
    "Hispa": {
        "desc": "Bệnh do sâu Hispa gây ra, lá bị cắn phá tạo vết trắng.",
        "solution": "Dùng thuốc trừ sâu, vệ sinh đồng ruộng, diệt trừ ký chủ phụ.",
        "color": (255, 0, 0)  # Màu xanh dương cho Hispa
    },
    "LeafBlast": {
        "desc": "Bệnh đạo ôn lá gây cháy lá, giảm năng suất.",
        "solution": "Phun thuốc đặc trị đạo ôn, bón phân hợp lý, không bón thừa đạm.",
        "color": (0, 255, 255)  # Màu vàng cho đạo ôn
    }
}

HISTORY_FILE = 'history.json'

GEMINI_API_KEY = "AIzaSyBeSpcUqH0vDFuW7yIYhnSLmVmyuAt3ZVw"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

CHATBOT_HISTORY_FILE = "chatbot_history.json"

# Thêm các biến cấu hình cho upload
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_FILE_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt', 'xls', 'xlsx'}

def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_FILE_EXTENSIONS

def draw_detection(image, class_name, confidence):
    """Vẽ viền và thông tin bệnh lên ảnh"""
    height, width = image.shape[:2]
    
    # Lấy màu tương ứng với loại bệnh
    color = DISEASE_INFO[class_name]['color']
    
    # Vẽ viền bao quanh ảnh
    cv2.rectangle(image, (0, 0), (width, height), color, 3)
    
    # Tạo background cho text
    text = f"{class_name}: {confidence:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    
    # Tính toán kích thước text
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Vẽ background cho text
    cv2.rectangle(image, 
                 (10, 10), 
                 (20 + text_width, 40 + text_height), 
                 color, 
                 -1)
    
    # Vẽ text
    cv2.putText(image, 
                text, 
                (20, 40), 
                font, 
                font_scale, 
                (255, 255, 255), 
                thickness)
    
    return image

def save_history(image_b64, result, method="Không xác định"):
    entry = {
        'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'image': image_b64,
        'result': result,
        'method': method
    }
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
    else:
        history = []
    history.append(entry)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f)

def get_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_chatbot_history(question, answer):
    entry = {
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer
    }
    try:
        if os.path.exists(CHATBOT_HISTORY_FILE):
            with open(CHATBOT_HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = []
        history.append(entry)
        with open(CHATBOT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Lỗi khi lưu lịch sử chatbot: {e}")

class RiceDiseasePredictor:
    def __init__(self, model_path="best_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ['BrownSpot', 'Healthy', 'Hispa', 'LeafBlast']
        self.transform = A.Compose([
            A.Resize(380, 380),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Load model
        self.model = efficientnet_b2(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, len(self.classes))
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

    def validate_image(self, image):
        """Validate and enhance image quality"""
        # Kiểm tra độ sáng
        brightness = np.mean(image)
        if brightness < 40:  # Ảnh quá tối
            image = cv2.convertScaleAbs(image, alpha=1.5, beta=30)
        elif brightness > 200:  # Ảnh quá sáng
            image = cv2.convertScaleAbs(image, alpha=0.7, beta=-30)
        
        # Kiểm tra độ tương phản
        contrast = np.std(image)
        if contrast < 30:  # Độ tương phản thấp
            image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
        
        return image

    def enhance_image(self, image):
        """Enhance image quality with various techniques"""
        # Chuyển đổi màu
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Cân bằng sáng
        image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        image_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0])
        image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)
        
        # Giảm nhiễu
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        # Tăng độ tương phản
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        return image

    def predict_from_file(self, image_path):
        """Dự đoán từ file ảnh"""
        # Đọc ảnh
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Lưu ảnh sau tiền xử lý
        os.makedirs('debug_images', exist_ok=True)
        cv2.imwrite('debug_images/processed_upload.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # Tiền xử lý
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)
        # Dự đoán
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        return {
            'class': self.classes[predicted.item()],
            'confidence': confidence.item() * 100,
            'probabilities': {cls: prob.item() * 100 for cls, prob in zip(self.classes, probabilities[0])}
        }

    def predict_from_bytes(self, image_bytes):
        """Dự đoán từ bytes ảnh (tự động crop sát vùng lá lúa, nhận diện xanh, vàng, nâu, làm nét và cân bằng sáng)"""
        try:
            # Chuyển đổi bytes thành ảnh
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image = np.array(image)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            # Xanh lá
            mask_green = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
            # Vàng
            mask_yellow = cv2.inRange(hsv, np.array([15, 40, 40]), np.array([35, 255, 255]))
            # Nâu
            mask_brown = cv2.inRange(hsv, np.array([5, 40, 40]), np.array([20, 255, 255]))
            # Gộp mask
            mask = cv2.bitwise_or(mask_green, mask_yellow)
            mask = cv2.bitwise_or(mask, mask_brown)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            height, width = image.shape[:2]
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                expand_w = int(w * 0.2)
                expand_h = int(h * 0.2)
                x1 = max(0, x - expand_w // 2)
                y1 = max(0, y - expand_h // 2)
                x2 = min(width, x + w + expand_w // 2)
                y2 = min(height, y + h + expand_h // 2)
                image = image[y1:y2, x1:x2]
            else:
                crop_ratio = 0.6
                crop_w, crop_h = int(width * crop_ratio), int(height * crop_ratio)
                x1 = (width - crop_w) // 2
                y1 = (height - crop_h) // 2
                x2 = x1 + crop_w
                y2 = y1 + crop_h
                image = image[y1:y2, x1:x2]
            # Tăng độ nét cho vùng crop
            kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
            image = cv2.filter2D(image, -1, kernel)
            # Cân bằng sáng cho vùng crop
            img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
            # Lưu ảnh sau tiền xử lý (debug)
            os.makedirs('debug_images', exist_ok=True)
            cv2.imwrite('debug_images/processed_webcam.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # Tiền xử lý giống predict_from_file
            transformed = self.transform(image=image)
            image_tensor = transformed['image'].unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            return {
                'class': self.classes[predicted.item()],
                'confidence': confidence.item() * 100,
                'probabilities': {cls: prob.item() * 100 for cls, prob in zip(self.classes, probabilities[0])}
            }
        except Exception as e:
            print(f"Error in predict_from_bytes: {str(e)}")
            return {'error': f'Lỗi: {str(e)}'}

# Khởi tạo predictor
predictor = RiceDiseasePredictor()

@app.route('/')
def home():
    return render_template('index.html', esp32_ip=ESP32_IP)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Không tìm thấy file'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Không có file được chọn'})
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Dự đoán
        result = predictor.predict_from_file(filepath)
        
        # Thông tin bệnh
        result['info'] = DISEASE_INFO.get(result['class'], {})
        
        # Lưu lịch sử (ảnh base64)
        with open(filepath, 'rb') as f:
            image_b64 = base64.b64encode(f.read()).decode('utf-8')
        save_history(image_b64, result, method="Tải lên")
        
        # Xóa file sau khi dự đoán
        os.remove(filepath)
        
        return jsonify(result)

@app.route('/predict_esp32', methods=['POST'])
def predict_esp32():
    try:
        esp32_url = f"http://{ESP32_IP}"
        response = requests.get(esp32_url.rstrip('/') + '/capture', timeout=5)
        if response.status_code == 200:
            result = predictor.predict_from_bytes(response.content)
            if 'error' not in result:
                result['info'] = DISEASE_INFO.get(result['class'], {})
                # Lưu lịch sử với ảnh gốc base64
                image_b64 = base64.b64encode(response.content).decode('utf-8')
                save_history(image_b64, result, method="ESP32-CAM")
            return jsonify(result)
        else:
            return jsonify({'error': f'Không thể lấy ảnh từ ESP32-CAM (HTTP {response.status_code})'})
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Kết nối đến ESP32-CAM bị timeout'})
    except requests.exceptions.ConnectionError:
        return jsonify({'error': 'Không thể kết nối đến ESP32-CAM'})
    except Exception as e:
        return jsonify({'error': f'Lỗi: {str(e)}'})

@app.route('/predict_webcam', methods=['POST'])
def predict_webcam():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'Không tìm thấy dữ liệu ảnh'})
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        result = predictor.predict_from_bytes(image_bytes)
        if 'error' not in result:
            result['info'] = DISEASE_INFO.get(result['class'], {})
            # Lưu lịch sử với ảnh gốc base64
            save_history(image_data, result, method="Webcam")
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Lỗi: {str(e)}'})

@app.route('/history', methods=['GET'])
def history():
    return jsonify(get_history())

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    question = data.get('question', '')
    try:
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": question
                        }
                    ]
                }
            ]
        }
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=30)
        result = response.json()
        with open("gemini_log.txt", "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        if "candidates" in result and result["candidates"]:
            answer = result["candidates"][0]["content"]["parts"][0]["text"].strip()
            save_chatbot_history(question, answer)
            return jsonify({'answer': answer})
        elif "error" in result:
            answer = f"Lỗi Gemini: {result['error'].get('message', str(result['error']))}"
            save_chatbot_history(question, answer)
            return jsonify({'answer': answer})
        else:
            answer = f"Lỗi không xác định từ Gemini: {result}"
            save_chatbot_history(question, answer)
            return jsonify({'answer': answer})
    except Exception as e:
        answer = f"Lỗi khi gọi Gemini: {str(e)}"
        save_chatbot_history(question, answer)
        return jsonify({'answer': answer})

@app.route('/chatbot_history', methods=['GET'])
def chatbot_history():
    try:
        if os.path.exists(CHATBOT_HISTORY_FILE):
            with open(CHATBOT_HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
                # Thêm thông tin file cho mỗi entry
                for entry in history:
                    if entry.get('type') == 'file':
                        filepath = entry.get('filepath')
                        if filepath and os.path.exists(filepath):
                            entry['file_url'] = f'/uploads/{os.path.basename(filepath)}'
        else:
            history = []
        return jsonify(history)
    except Exception as e:
        return jsonify({'error': f'Lỗi khi đọc lịch sử chatbot: {str(e)}'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/chatbot_upload_image', methods=['POST'])
def chatbot_upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'Không tìm thấy file ảnh'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Không có file được chọn'})
    
    if file and allowed_image_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Chuyển ảnh thành base64
        with open(filepath, 'rb') as f:
            image_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Lưu vào lịch sử chat
        entry = {
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "image",
            "filename": filename,
            "image": image_b64
        }
        save_chatbot_history_entry(entry)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'image': image_b64
        })
    
    return jsonify({'error': 'File không được hỗ trợ'})

@app.route('/chatbot_upload_file', methods=['POST'])
def chatbot_upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Không tìm thấy file'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Không có file được chọn'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Lưu vào lịch sử chat
        entry = {
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "file",
            "filename": filename,
            "filepath": filepath
        }
        save_chatbot_history_entry(entry)
        
        return jsonify({
            'success': True,
            'filename': filename
        })
    
    return jsonify({'error': 'File không được hỗ trợ'})

def save_chatbot_history_entry(entry):
    try:
        if os.path.exists(CHATBOT_HISTORY_FILE):
            with open(CHATBOT_HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = []
        history.append(entry)
        with open(CHATBOT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Lỗi khi lưu lịch sử chatbot: {e}")

@app.route('/delete_history/<int:index>', methods=['DELETE'])
def delete_history(index):
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
            
            if 0 <= index < len(history):
                # Xóa file ảnh nếu có
                image_data = history[index].get('image')
                if image_data:
                    try:
                        # Chuyển base64 thành file tạm để kiểm tra
                        image_bytes = base64.b64decode(image_data)
                        image = Image.open(io.BytesIO(image_bytes))
                        
                        # Kiểm tra xem có phải ảnh có mặt người không
                        # (Bạn có thể thêm logic kiểm tra ở đây)
                        
                        # Xóa entry khỏi lịch sử
                        history.pop(index)
                        
                        # Lưu lại lịch sử
                        with open(HISTORY_FILE, 'w') as f:
                            json.dump(history, f)
                            
                        return jsonify({'success': True, 'message': 'Đã xóa ảnh thành công'})
                    except Exception as e:
                        return jsonify({'error': f'Lỗi khi xử lý ảnh: {str(e)}'})
            
            return jsonify({'error': 'Không tìm thấy ảnh cần xóa'})
        return jsonify({'error': 'Không tìm thấy file lịch sử'})
    except Exception as e:
        return jsonify({'error': f'Lỗi khi xóa ảnh: {str(e)}'})

@app.route('/delete_history_by_time/<time>', methods=['DELETE'])
def delete_history_by_time(time):
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
            # Tìm entry có time khớp
            new_history = [entry for entry in history if entry.get('time') != time]
            if len(new_history) == len(history):
                return jsonify({'error': 'Không tìm thấy ảnh cần xóa'})
            with open(HISTORY_FILE, 'w') as f:
                json.dump(new_history, f)
            return jsonify({'success': True, 'message': 'Đã xóa ảnh thành công'})
        return jsonify({'error': 'Không tìm thấy file lịch sử'})
    except Exception as e:
        return jsonify({'error': f'Lỗi khi xóa ảnh: {str(e)}'})

if __name__ == '__main__':
    print("\n* Running on:")
    print("  - http://127.0.0.1:5000/ (localhost)")
    print("  - http://192.168.1.25:5000/ (network)\n")
    app.run(host='0.0.0.0', port=5000, debug=True) 