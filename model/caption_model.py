from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Add
from tensorflow.keras.models import Model

def caption_model(vocab_size, max_length):
    # Đầu vào đặc trưng ảnh
    image_input = Input(shape=(2048,), name="image_input")
    image_dense = Dense(256, activation="relu")(image_input)
    
    # Đầu vào chuỗi văn bản
    text_input = Input(shape=(max_length,), name="text_input")
    text_embedding = Embedding(vocab_size, 256, mask_zero=True)(text_input)
    text_lstm = LSTM(256)(text_embedding)
    
    # Kết hợp hai đầu vào
    decoder = Add()([image_dense, text_lstm])
    output = Dense(vocab_size, activation="softmax")(decoder)
    
    model = Model(inputs=[image_input, text_input], outputs=output)
    return model