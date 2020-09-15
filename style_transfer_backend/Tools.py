from numba import cuda
from scipy.optimize import fmin_l_bfgs_b
from datetime import datetime
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from PIL import Image
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.models import Sequential
from keras.applications.vgg19 import VGG19

import os
import numpy as np
import PIL.Image
import tensorflow as tf
from style_transfer_backend import Config as conf
import matplotlib.pyplot as plt


def unpreprocess(img):
    img[..., 0] += 103.939
    img[..., 1] += 116.779
    img[..., 2] += 126.68
    img = img[..., ::-1]
    return img


def scale_img(x):
    x = x - x.min()
    x = x / x.max()
    return x


def load_img_and_preprocess_resize(path, resize=512):
    img = resize_image(Image.open(path), resize)
    return convert_image_to_array_vgg(img)


def load_img_and_preprocess_shape(path, shape=None):
    img = image.load_img(path, target_size=shape)
    return convert_image_to_array_vgg(img)


def convert_image_to_array_vgg(img):
    # convert image to array and preprocess for vgg
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x


def resize_image(img, max_size=512):
    """:param
    This function resize the image to max_size in width or height by
    conserving the scale between height and width
    """
    img.thumbnail((max_size, max_size))
    return img


def clean_gpu_memory():
    device = cuda.get_current_device()
    device.reset()


def create_clear_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        filelist = [f for f in os.listdir(path)]
        for f in filelist:
            os.remove(os.path.join(path, f))


def LBFGS_Optimizer(fn, epochs, batch_shape):
    t0 = datetime.now()
    losses = []
    x = np.random.randn(np.prod(batch_shape))
    for i in range(epochs):
        x, l, _ = fmin_l_bfgs_b(
            func=fn,
            x0=x,
            maxfun=20
        )
        x = np.clip(x, -127, 127)
        print("iter=%s, loss=%s" % (i, l))
        losses.append(l)

    print("duration:", datetime.now() - t0)
    # plt.plot(losses)
    # plt.show()

    newimg = x.reshape(*batch_shape)
    final_img = unpreprocess(newimg)
    return final_img[0], losses


def VGG19_AvgPool(shape):
    """:param
         Reconstruction of VGG19
    """
    # 1: remove the last three layers, because they are for classification tasks
    vgg = VGG19(input_shape=shape, weights='imagenet', include_top=False)

    # 2: replace all Max-Pool Layers with the AVG-Pool Layer
    new_model = Sequential()
    for layer in vgg.layers:
        if layer.__class__ == MaxPooling2D:
            # replace it with average pooling
            new_model.add(AveragePooling2D())
        else:
            new_model.add(layer)

    return new_model


def tensor_to_image(tensor):
    """:param
    It is a post process function which convert the vector to an image
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def load_img(path_to_img, resize):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)

    scale = conf.max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    if resize:
        img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]

    return img


def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


def high_pass_x_y(image):
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

    return x_var, y_var


def draw_high_noise(image, content_image):
    x_deltas, y_deltas = high_pass_x_y(content_image)

    plt.figure(figsize=(14, 10))
    plt.subplot(2, 2, 1)
    imshow(clip_0_1(2 * y_deltas + 0.5), "Horizontal Deltas: Original")

    plt.subplot(2, 2, 2)
    imshow(clip_0_1(2 * x_deltas + 0.5), "Vertical Deltas: Original")

    x_deltas, y_deltas = high_pass_x_y(image)

    plt.subplot(2, 2, 3)
    imshow(clip_0_1(2 * y_deltas + 0.5), "Horizontal Deltas: Styled")

    plt.subplot(2, 2, 4)
    imshow(clip_0_1(2 * x_deltas + 0.5), "Vertical Deltas: Styled")


def get_file_name_to_save_result(directory):
    """:param
    This function prepares the file name to save the result,
    keep in mind that the file name does not have extension
    """
    import random
    ran_num = random.randrange(1, 100, 3)
    file_name = "{num}_{content_lay}_{content_factor}_{style_factor}_{noise}_{epo}".format(
        num=str(ran_num),
        content_lay=conf.content_layers[0],
        content_factor=conf.content_weight,
        style_factor=conf.style_weight,
        noise=conf.total_variation_weight,
        epo=conf.epochs * conf.steps_per_epoch
    )
    file_name = directory + "/" + file_name
    return file_name
