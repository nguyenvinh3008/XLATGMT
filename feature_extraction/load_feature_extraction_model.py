from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model

#tải mô hình trích xuất đặc trưng đã huấn luyện trước.
def load_feature_extraction_model():
    """
    Load and return a pre-trained InceptionV3 model for feature extraction.
    """
    # Load InceptionV3 model pre-trained on ImageNet
    model_incep = InceptionV3(weights='imagenet')

    # Remove the final classification layer to use the output of 'avg_pool' as feature vector
    model_incep_new = Model(inputs=model_incep.input, outputs=model_incep.get_layer('avg_pool').output)

    return model_incep_new
