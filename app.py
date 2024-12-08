from flask import Flask, render_template, request, jsonify, send_from_directory
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import numpy as np
import os
import json
from feature_extraction.feature_extraction import preprocess_image, extract_features
from feature_extraction.load_feature_extraction_model import load_feature_extraction_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
from tensorflow.keras.preprocessing.text import tokenizer_from_json

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

app = Flask(__name__)

# Đặt thư mục upload là thư mục tĩnh để Flask có thể truy cập và phục vụ ảnh
app.config['UPLOAD_FOLDER'] = 'upload'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Tối đa 16MB cho file tải lên

# Tạo thư mục nếu chưa có
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Đường dẫn tới mô hình và tokenizer
model_path = r'D:\test\results\model.keras'  # Thay đổi từ model.keras sang model.h5
weights_path = r'D:\test\results\model_weights.weights.h5'
tokenizer_path = r'D:\test\tokenizer.json'

# Tải mô hình và trọng số
try:
    model = load_model(model_path)  # Sử dụng .h5 thay vì .keras
    model.load_weights(weights_path)
    print("Mô hình và trọng số đã được tải thành công.")
except Exception as e:
    print(f"Lỗi khi tải mô hình hoặc trọng số: {e}")
    model = None  # Đảm bảo model được đặt là None nếu có lỗi

# Sửa đoạn tải tokenizer
try:
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer = tokenizer_from_json(json.load(f))  # Sử dụng tokenizer_from_json
    print("Tokenizer đã được tải thành công.")
except Exception as e:
    print(f"Lỗi khi tải tokenizer: {e}")
    tokenizer = None  # Đảm bảo tokenizer được đặt là None nếu có lỗi

# Hàm tạo caption cho ảnh
def generate_caption(image_path, max_length=33):  
    if model is None or tokenizer is None:
        return "Error: Model or Tokenizer is not loaded properly."

    try:
        # Trích xuất đặc trưng từ ảnh
        image_features = extract_features(image_path, load_feature_extraction_model())
        image_features = np.reshape(image_features, (1, 2048))  # Đảm bảo kích thước (1, 2048)

        # Chuyển đổi câu bắt đầu thành chuỗi số
        input_sequence = tokenizer.texts_to_sequences(['<start>'])[0]
        
        # Đảm bảo chiều dài của input_sequence không vượt quá max_length
        if len(input_sequence) > max_length:
            input_sequence = input_sequence[:max_length]

        input_sequence = pad_sequences([input_sequence], maxlen=max_length, padding='post')  # Điều chỉnh kích thước sequence
        input_sequence = np.reshape(input_sequence, (1, max_length))  # Đảm bảo kích thước (1, max_length)

        caption = []
        for i in range(max_length):
            # Kiểm tra kích thước đầu vào mỗi bước
            print(f"Step {i + 1}:")
            print(f"image_features shape: {image_features.shape}")
            print(f"input_sequence shape: {input_sequence.shape}")

            # Dự đoán tiếp theo
            yhat = model.predict([image_features, input_sequence], verbose=0)
            print(f"Prediction output: {yhat}")

            # Áp dụng softmax cho phân phối xác suất (nếu cần)
            yhat = np.exp(yhat) / np.sum(np.exp(yhat))  # Chuyển đổi thành phân phối xác suất
            print(f"Softmax prediction: {yhat}")

            # Chọn từ ngẫu nhiên dựa trên phân phối xác suất
            yhat = np.random.choice(np.arange(len(yhat[0])), p=yhat[0])  # Chọn từ ngẫu nhiên
            print(f"Predicted word index: {yhat}")

            word = tokenizer.index_word.get(yhat, '')  # Lấy từ từ index
            print(f"Predicted word: {word}")

            if word == '<end>':
                break
            caption.append(word)
            
            # Cập nhật input_sequence cho bước tiếp theo
            input_sequence = np.append(input_sequence, yhat)
            input_sequence = input_sequence[1:]  # Giới hạn chiều dài sequence về max_length
            
            # Cập nhật lại input_sequence thành dạng (1, max_length)
            input_sequence = np.reshape(input_sequence, (1, max_length))

        caption_text = ' '.join(caption)

        # Lưu caption vào file "captions.txt"
        with open('captions.txt', 'a', encoding='utf-8') as file:
            file.write(f"Image: {image_path}\n")
            file.write(f"Caption: {caption_text}\n\n")

        return caption_text
    except Exception as e:
        print(f"Lỗi khi tạo caption cho ảnh {image_path}: {e}")
        return "Error generating caption"

# Route để phục vụ ảnh từ thư mục upload
@app.route('/upload/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Trang chủ với form upload ảnh
@app.route('/')
def index():
    return render_template('index.html')

# Xử lý ảnh tải lên và tạo caption
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    image = request.files['image']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(image_path)

    # Tạo caption cho ảnh
    max_length = 33  # Độ dài tối đa của caption
    caption = generate_caption(image_path, max_length)

    # Trả về URL của ảnh và caption
    image_url = f'/upload/{image.filename}'  # Đường dẫn tĩnh đến ảnh
    return render_template('index.html', caption=caption, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
