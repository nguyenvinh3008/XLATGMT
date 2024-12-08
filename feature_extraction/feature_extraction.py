from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
import tensorflow as tf
import numpy as np

def preprocess_image(image_path, target_size=(299, 299)):
    # Tải ảnh từ đường dẫn và thay đổi kích thước
    image = load_img(image_path, target_size=target_size)  # Resize ảnh thành kích thước yêu cầu
    image = img_to_array(image)  # Chuyển ảnh thành mảng numpy
    image = np.expand_dims(image, axis=0)  # Thêm chiều batch size
    image = preprocess_input(image)  # Tiền xử lý ảnh theo chuẩn của InceptionV3
    return image

# Hàm trích xuất đặc trưng từ ảnh
def extract_features(image_path, model):
    # Tiền xử lý ảnh
    image = preprocess_image(image_path, target_size=(299, 299))  # Đảm bảo kích thước ảnh hợp lệ và tiền xử lý
    features = model.predict(image)  # Dự đoán và trích xuất đặc trưng
    features = np.reshape(features, (features.shape[1]))  # Làm phẳng đặc trưng
    return features
