from __future__ import print_function, division
# In this script, we will focus on generating an image
# that attempts to match the content of one input image
# and the style of another input image.
#
# We accomplish this by balancing the content loss
# and style loss simultaneously.

from style_transfer_backend import Tools, ContentReconstruction, StyleReconstruction

import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)


def style_transfer(content_image_path, style_image_path):
    mydir = './Outputs/style_transfer_details'

    # 1: load the content and style images, then rescale the style image to the scale of content image
    content_img = Tools.load_img_and_preprocess_resize(content_image_path, resize=512)
    h, w = content_img.shape[1:3]

    # test_content_reconstruction(content_img[0], mydir, [16, 17, 18, 19], 0)

    style_img = Tools.load_img_and_preprocess_shape(style_image_path, (h, w))
    # show all blocks output
    # test_style_reconstruction(style_img[0], mydir, 1)

    batch_shape = content_img.shape
    shape = content_img.shape[1:]

    vgg = Tools.VGG19_AvgPool(shape)
    print(vgg.summary())

    # 2: get content and style features + features extractor model
    content_features, content_features_extractor_model = ContentReconstruction.get_content_image_features(content_img,
                                                                                                          14, vgg)
    style_layers_features_outputs, symbolic_conv_outputs, style_features_extractor_model = StyleReconstruction.get_style_image_features(
        style_img, 5, vgg)

    # we will assume the weight of the content loss is 1
    # and only weight the style losses
    style_weights = [0.2, 0.4, 0.3, 0.5, 0.2]
    # style_weights = [0.4, 0.6, 0.6, 0.7, 0.4]

    # create the total loss which is the sum of content + style loss
    loss = 1 * K.mean(K.square(content_features_extractor_model.output - content_features))

    for w, symbolic, actual in zip(style_weights, symbolic_conv_outputs, style_layers_features_outputs):
        # gram_matrix() expects a (H, W, C) as input
        loss += w * Tools.style_loss(symbolic[0], actual[0])

    # loss += 0.0001 * tf.image.total_variation(vgg.input)

    # once again, create the gradients and loss + grads function
    # note: it doesn't matter which model's input you use
    # they are both pointing to the same keras Input layer in memory
    grads = K.gradients(loss, vgg.input)

    # just like theano.function
    get_loss_and_grads = K.function(
        inputs=[vgg.input],
        outputs=[loss] + grads
    )

    def get_loss_and_grads_wrapper(x_vec):
        l, g = get_loss_and_grads([x_vec.reshape(*batch_shape)])
        return l.astype(np.float64), g.flatten().astype(np.float64)

    final_image, losses = Tools.LBFGS_Optimizer(get_loss_and_grads_wrapper, 10, batch_shape)

    # plot loss
    plt.plot(losses)
    # plt.savefig(plot_name)
    plt.show()

    # save image
    final_image = Tools.scale_img(final_image)
    plt.imshow(final_image)
    # plt.imsave(file_name, final_image)
    plt.show()
