from tensorflow.keras.models import load_model
import tensorflow as tf
print(tf.__version__)

model = load_model('EfficientNet_B0.h5')
print(model.summary())