from __future__ import print_function, division

from Tools import scale_img, LBFGS_Optimizer, convert_image_to_array_vgg, VGG19_AvgPool

from keras.models import Model
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)


def get_content_image_features(content_image, output_layer, vgg_model):
    """:param
       This function extracts the content image features from the specified layer
       output_layer: is the convolution layer number : 2,3,4,5,6,7,8,9,10,11,12,13,14
    """
    # 1: create content features extractor
    content_model = Model(vgg_model.input, vgg_model.layers[output_layer].get_output_at(0))

    # 2: get content image features
    target = K.variable(content_model.predict(content_image))

    return target, content_model


def reconstruct_content_image(img, conv_layer, file_name, plot_name):
    """:param
    This function reconstructs the input image from the given convolution layer number
    in VGG19 architecture
    """
    # convert image to array and preprocess for vgg
    x = convert_image_to_array_vgg(img)

    # we'll use this throughout the rest of the script
    batch_shape = x.shape
    shape = x.shape[1:]
    #
    # # see the image
    # plt.imshow(img)
    # plt.show()

    # make a content model
    # try different cutoffs to see the images that result
    vgg = VGG19_AvgPool(shape)
    content_features, content_model_extractor = get_content_image_features(x, conv_layer, vgg)

    # define our loss in keras
    loss = K.mean(K.square(content_features - content_model_extractor.output))

    # gradients which are needed by the optimizer
    grads = K.gradients(loss, content_model_extractor.input)

    # just like theano.function
    get_loss_and_grads = K.function(
        inputs=[content_model_extractor.input],
        outputs=[loss] + grads
    )

    def get_loss_and_grads_wrapper(x_vec):
        # scipy's minimizer allows us to pass back
        # function value f(x) and its gradient f'(x)
        # simultaneously, rather than using the fprime arg
        #
        # we cannot use get_loss_and_grads() directly
        # input to minimizer func must be a 1-D array
        # input to get_loss_and_grads must be [batch_of_images]
        #
        # gradient must also be a 1-D array
        # and both loss and gradient must be np.float64
        # will get an error otherwise

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


def test_content_reconstruction(content_image, mydir, layers, image_number):
    for layer in layers:
        reconstruct_content_image(content_image,
                                  layer,
                                  mydir + "/reconstruction_{number}_conv_{l_number}.jpg".format(number=image_number,
                                                                                                l_number=layer),
                                  mydir + "/reconstruction_{number}_conv_{l_number}_loss.jpg".format(
                                      number=image_number,
                                      l_number=layer))
