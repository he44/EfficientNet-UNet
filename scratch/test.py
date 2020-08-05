import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model
import tensorflow as tf

print(tf.__version__)

dev_tmp = r'./../tmp'
if not os.path.isdir(dev_tmp):
    os.mkdir(dev_tmp)

pre_trained_ckpt = False
feature_extraction = True

"""
Running pre-trained checkpoints (good)
"""
if pre_trained_ckpt:
    # Following this notebook: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/eval_ckpt_example.ipynb
    import eval_ckpt_main as eval_ckpt

    model_name = 'efficientnet-b0'
    ckpt_dir = model_name
    eval_driver = eval_ckpt.get_eval_driver(model_name)
    print(type(eval_driver))

    image_file = 'labrador.jpg'
    image_files = [image_file]

    labels_map_file = 'labels_map.json'

    pred_idx, pred_prob = eval_driver.eval_example_images(
        ckpt_dir, image_files, labels_map_file
    )

    print(pred_idx, pred_prob)

"""
Using EfficientNet as Feature Extractor
"""
if feature_extraction:
    import efficientnet_builder
    import numpy as np

    """
    # random input
    input_name = 'random'
    # images = np.random.random(size=(1, 224, 224, 3))
    images = np.random.random(size=(1, 1024, 1024, 3))
    """

    # input as panda
    #input_name = 'panda'
    input_name = 'labrador'
    import imageio
    images = imageio.imread("%s.jpg"%input_name)
    images = np.reshape(images, (1, images.shape[0], images.shape[1], images.shape[2]))
    images = images.astype(np.float32)
    print(images.shape, images.dtype)

    images = tf.convert_to_tensor(images)
    print(images.shape, images.dtype)
    model_name = 'efficientnet-b0'
    features, endpoints = efficientnet_builder.build_model_base(
        images=images,
        model_name=model_name,
        training=True
    )

    print(type(features))
    print(type(endpoints))

    endpoint_tensors = []
    for i in range(1, 6):
        endpoint_tensors.append(endpoints['reduction_%d' % i])

    endpoint_numpys = []

    sess = tf.Session()
    with sess.as_default():
        init = tf.global_variables_initializer()
        sess.run(init)
        features_numpy = features.eval()
        print(type(features_numpy))
        print(features_numpy.shape, features_numpy.dtype)

        for endpoint_tensor in endpoint_tensors:
            endpoint_numpy = endpoint_tensor.eval()
            print(endpoint_numpy.shape)
            endpoint_numpys.append(endpoint_numpy)

    # let's save some features and see what they are
    import random
    import matplotlib.pyplot as plt

    random_c = random.sample(range(320), 16)
    print(random_c)

    fig, axs = plt.subplots(4, 4)
    fig.set_size_inches(12, 12)
    for rr in range(4):
        for cc in range(4):
            axs[rr, cc].imshow(features_numpy[0, :, :, random_c[rr * 4 + cc]])
            #axs[rr, cc].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(dev_tmp, '%s_features.png' % input_name))
    plt.clf()

    for i in range(1, len(endpoint_numpys) + 1):
        endpoint_numpy = endpoint_numpys[i - 1]
        plt.imshow(endpoint_numpy[0, :, :, 0])
        plt.savefig(os.path.join(dev_tmp, '%s_first_map_in_endpoint_%d.png' % (input_name, i)))
        plt.clf()
