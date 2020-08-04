import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model
import tensorflow as tf
print(tf.__version__)

"""
model = load_model('EfficientNet_B0.h5')
print(model.summary())
"""

"""
model_name = 'efficientnet-b0'
ckpt_dir = model_name
checkpoint = tf.train.latest_checkpoint(ckpt_dir)
print(type(checkpoint), checkpoint)
"""

# Following this notebook: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/eval_ckpt_example.ipynb
import eval_ckpt_main as eval_ckpt
model_name = 'efficientnet-b0'
ckpt_dir = model_name
eval_driver = eval_ckpt.get_eval_driver(model_name)
print(type(eval_driver))

image_file = 'panda.jpg'
image_files = [image_file]

labels_map_file = 'labels_map.json'

pred_idx, pred_prob = eval_driver.eval_example_images(
    ckpt_dir, image_files, labels_map_file
)

print(pred_idx, pred_prob)