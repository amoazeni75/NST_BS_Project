from __future__ import print_function, division

# In this script, we will focus on generating an image
# with the same style as the input image.
# But NOT the same content.
# It should capture only the essence of the style.

from keras.models import Model
from Tools import scale_img, LBFGS_Optimizer, convert_image_to_array_vgg, VGG19_AvgPool

import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K

np.random.seed(7)


def gram_matrix(img):
    # input is (H, W, C) (C = # feature maps)
    # we first need to convert it to (C, H*W)
    X = K.batch_flatten(K.permute_dimensions(img, (2, 0, 1)))

    # now, calculate the gram matrix
    # gram = XX^T / N
    # the constant is not important since we'll be weighting these
    G = K.dot(X, K.transpose(X)) / img.get_shape().num_elements()
    return G


def style_loss(y, t):
    return K.mean(K.square(gram_matrix(y) - gram_matrix(t)))


def get_style_image_features(style_image, number_of_participated_block, vgg_model):
    """:param
    This function extracts the style image features based on the number of
    blocks that user wants to participate. Also, in this feature extraction process
    we consider the output of the first convolution layer in each block
    number_of_participated_block: 1,2,3,4,5
    """
    # 1 get the basic VGG19 network

    # 2: select the output of the first convolution layer in each block
    symbolic_conv_outputs = [
        layer.get_output_at(1) for layer in vgg_model.layers \
        if layer.name.endswith('conv1')
    ]

    # 3: select the outputs based on the number of participated blocks
    symbolic_conv_outputs = symbolic_conv_outputs[0:number_of_participated_block]

    # 4: make a big model that outputs multiple layers' outputs
    multi_output_model = Model(vgg_model.input, symbolic_conv_outputs)

    # 5: calculate the targets that are output at each layer
    style_layers_outputs = [K.variable(y) for y in multi_output_model.predict(style_image)]

    return style_layers_outputs, symbolic_conv_outputs, multi_output_model


def reconstruct_style_image(img, conv_layer, file_name, plot_name):
    # convert image to array and preprocess for vgg
    x = convert_image_to_array_vgg(img)

    # we'll use this throughout the rest of the script
    batch_shape = x.shape
    shape = x.shape[1:]

    vgg = VGG19_AvgPool(shape)
    style_layers_features_outputs, symbolic_conv_outputs, style_features_extractor_model = get_style_image_features(x,
                                                                                                                    conv_layer,
                                                                                                                    vgg)

    # calculate the total style loss
    loss = 0
    for symbolic, actual in zip(symbolic_conv_outputs, style_layers_features_outputs):
        # gram_matrix() expects a (H, W, C) as input
        if conv_layer == 1:
            loss += style_loss(symbolic[0], actual)
        else:
            loss += style_loss(symbolic[0], actual[0])

    grads = K.gradients(loss, style_features_extractor_model.input)

    # just like theano.function
    get_loss_and_grads = K.function(
        inputs=[style_features_extractor_model.input],
        outputs=[loss] + grads
    )

    def get_loss_and_grads_wrapper(x_vec):
        l, g = get_loss_and_grads([x_vec.reshape(*batch_shape)])
        return l.astype(np.float64), g.flatten().astype(np.float64)

    final_image, losses = LBFGS_Optimizer(get_loss_and_grads_wrapper, 10, batch_shape)

    # plot loss
    plt.plot(losses)
    plt.savefig(plot_name)
    plt.show()

    # save image
    final_image = scale_img(final_image)
    plt.imshow(final_image)
    plt.imsave(file_name, final_image)
    plt.show()


def test_style_reconstruction(style_image, mydir, image_number):
    blocks = [1, 2, 3, 4, 5]

    for block_num in blocks:
        reconstruct_style_image(style_image,
                                block_num,
                                mydir + "/reconstruction_{number}_conv_{l_number}.jpg".format(number=image_number,
                                                                                              l_number=block_num),
                                mydir + "/reconstruction_{number}_conv_{l_number}_loss.jpg".format(number=image_number,
                                                                                                   l_number=block_num))
