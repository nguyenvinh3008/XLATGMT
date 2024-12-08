#huấn luyện mô hình học sâu (deep learning model)
import json
import os
import numpy as np
import tensorflow as tf
from model.caption_model import caption_model
from feature_extraction.feature_extraction import preprocess_image, extract_features
from feature_extraction.load_feature_extraction_model import load_feature_extraction_model
import sys
 
# Thêm thư mục hiện tại vào sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Tạo thư mục lưu kết quả nếu chưa có
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# Thiết lập stdout để in ký tự Unicode
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

# Đọc captions từ tệp
def load_captions(captions_file):
    caption_dict = {}
    if not os.path.exists(captions_file):
        print(f"Lỗi: Tệp {captions_file} không tồn tại!")
        return caption_dict

    try:
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    image_filename = parts[0].strip()
                    caption = parts[1].strip()
                    img_path = os.path.normpath(os.path.join(r'D:\test\dataset\Images', image_filename))
                    caption_dict[img_path] = caption
    except Exception as e:
        print(f"Lỗi khi đọc tệp captions: {e}")
    
    return caption_dict

# Load captions
captions_file = r'D:\test\dataset\captions.txt'
caption_dict = load_captions(captions_file)

# Xử lý ảnh không tìm thấy
missing_images_log = os.path.join(results_dir, 'missing_images.log')
found_images_log = os.path.join(results_dir, 'found_images.log')

with open(missing_images_log, 'w', encoding='utf-8') as missing_log, open(found_images_log, 'w', encoding='utf-8') as found_log:
    missing_images = [img for img in caption_dict.keys() if not os.path.exists(img)]
    if missing_images:
        missing_log.write("\n".join(missing_images))
        print(f"Không tìm thấy {len(missing_images)} ảnh. Danh sách lưu tại: {missing_images_log}")
        caption_dict = {k: v for k, v in caption_dict.items() if k not in missing_images}
    else:
        print("Tất cả ảnh đã được tìm thấy.")

    found_images = [img for img in caption_dict.keys()]
    found_log.write("\n".join(found_images))
    print(f"Đã tìm thấy {len(found_images)} ảnh. Danh sách lưu tại: {found_images_log}")

# Kiểm tra caption
if not caption_dict:
    print("Không có captions hợp lệ. Kết thúc chương trình.")
    sys.exit()

# Tiền xử lý caption
captions = list(caption_dict.values())
captions = ['<start> ' + caption + ' <end>' for caption in captions]

# Tạo tokenizer và huấn luyện
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts(captions)

# Đảm bảo <start> và <end> trong word_index
if '<start>' not in tokenizer.word_index:
    tokenizer.word_index['<start>'] = len(tokenizer.word_index) + 1
if '<end>' not in tokenizer.word_index:
    tokenizer.word_index['<end>'] = len(tokenizer.word_index) + 1

# Kiểm tra word_index
print("Kiểm tra word_index của tokenizer:")
print(tokenizer.word_index)

# Độ dài tối đa
max_length = max(len(caption.split()) for caption in captions) + 1
print(f"Độ dài caption tối đa: {max_length}")

# Lưu tokenizer vào JSON
tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w', encoding='utf-8') as f:
    json.dump(tokenizer_json, f, ensure_ascii=False, indent=4)
print("Tokenizer đã được lưu vào 'tokenizer.json'.")

# Mô hình Encoder-Decoder
def build_model(vocab_size, max_length):
    input_image_features = tf.keras.Input(shape=(2048,))
    image_features = tf.keras.layers.Dense(256, activation='relu')(input_image_features)

    input_sequence = tf.keras.Input(shape=(max_length - 1,))
    embedding = tf.keras.layers.Embedding(vocab_size, 256)(input_sequence)
    lstm = tf.keras.layers.LSTM(256)(embedding)

    decoder_input = tf.keras.layers.add([image_features, lstm])
    output = tf.keras.layers.Dense(vocab_size, activation='softmax')(decoder_input)

    model = tf.keras.Model(inputs=[input_image_features, input_sequence], outputs=output)
    return model

# Xây dựng mô hình
model = build_model(len(tokenizer.word_index) + 1, max_length - 1)

# Trích xuất đặc trưng từ ảnh
feature_model = load_feature_extraction_model()
image_features = []

for img_path in caption_dict.keys():
    try:
        features = extract_features(img_path, feature_model)
        features = np.reshape(features, (-1,))
        if features.shape != (2048,):
            features = np.zeros((2048,))
    except Exception as e:
        features = np.zeros((2048,))
    image_features.append(features)

image_features = np.array(image_features)
print(f"Kích thước image_features: {image_features.shape}")

# Chuyển đổi captions thành số
sequences = tokenizer.texts_to_sequences(captions)
sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding='post')

if len(image_features) != sequences.shape[0]:
    print("Kích thước không khớp giữa image_features và sequences.")
    sys.exit()

# Chuẩn bị dữ liệu
sequences_input = sequences[:, :-1]
sequences_output = sequences[:, 1:]

# Huấn luyện
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

model_path = os.path.join(results_dir, 'model.keras')
weights_path = os.path.join(results_dir, 'model_weights.weights.h5')

if os.path.exists(model_path):
    os.remove(model_path)

if os.path.exists(weights_path):
    os.remove(weights_path)

model.build(input_shape=[(None, 2048), (None, max_length - 1)])
model.save(model_path)
model.save_weights(weights_path)

print(f"Mô hình đã lưu: {model_path}")
print(f"Trọng số đã lưu: {weights_path}")
