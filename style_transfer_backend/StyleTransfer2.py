from style_transfer_backend import Tools
from style_transfer_backend import Config
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import time

# variables
num_content_layers = -1
num_style_layers = -1
extractor = ""  # is the object that ables to extract the style from content
opt = ""  # is the optimizer object
style_targets = ""  # is the input style image
content_targets = ""  # is the input content images


def get_stylized_image_from_fast_style_transfer(content_image, style_image):
    hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
    return stylized_image


def get_VGG19():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    if Config.use_avg_pool_layers:
        new_model = tf.keras.Sequential()
        for layer in vgg.layers:
            if layer.__class__ == tf.keras.layers.MaxPooling2D:
                # replace it with average pooling
                new_model.add(tf.keras.layers.AveragePooling2D())
            else:
                new_model.add(layer)

        return new_model
    else:
        return vgg


def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    vgg = get_VGG19()
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [Tools.gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                           for name in style_outputs.keys()])
    style_loss *= Config.style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                             for name in content_outputs.keys()])
    content_loss *= Config.content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss


def total_variation_loss(image):
    x_deltas, y_deltas = Tools.high_pass_x_y(image)
    return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))


def train_step(image, losses):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        loss += Config.total_variation_weight * tf.image.total_variation(image)
        if losses is not None:
            losses.append(loss.numpy()[0])

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(Tools.clip_0_1(image))


def do_style_transfer(content_image, style_image):
    global num_content_layers
    global num_style_layers
    global extractor
    global opt
    global style_targets
    global content_targets

    num_content_layers = len(Config.content_layers)
    num_style_layers = len(Config.style_layers)

    # 1: initiate style and content extractor
    extractor = StyleContentModel(Config.style_layers, Config.content_layers)
    results = extractor(tf.constant(content_image))

    # 2: print output details
    if Config.show_log:
        print('Styles:')
        for name, output in sorted(results['style'].items()):
            print("  ", name)
            print("    shape: ", output.numpy().shape)
            print("    min: ", output.numpy().min())
            print("    max: ", output.numpy().max())
            print("    mean: ", output.numpy().mean())
            print()
        print("Contents:")
        for name, output in sorted(results['content'].items()):
            print("  ", name)
            print("    shape: ", output.numpy().shape)
            print("    min: ", output.numpy().min())
            print("    max: ", output.numpy().max())
            print("    mean: ", output.numpy().mean())

    # 3: get separate output for style and content, the style_targets and content_targets are
    # actually the input images
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    # 4: define output image
    # to make this quick, initialize it with the content image instead of white noise image
    output_image = tf.Variable(content_image)

    # 5: define adam optimizer
    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    # 6: draw the high frequency noises in the inputs
    Tools.draw_high_noise(output_image, content_image)
    plt.show()

    # 7: Start Optimization
    start = time.time()
    step = 0
    losses = []
    for n in range(Config.epochs):
        temp = []
        for m in range(Config.steps_per_epoch):
            step += 1
            train_step(output_image, temp)
            if temp is not None:
                losses.extend(temp)
                temp = None
            print(".", end='')
        intermediate_image = Tools.tensor_to_image(output_image)
        plt.imshow(intermediate_image)
        plt.show()

        print("Train step: {}".format(step))

    end = time.time()
    print("Total time: {:.1f} minutes".format((end - start) / 60))

    return output_image, losses


def style_transfer(c_image=None, s_image=None):
    if Config.show_log:
        # 1: create output directory to save the result
        directory = './Outputs/StyleTransfer_V2_Results'
        file_name = Tools.get_file_name_to_save_result(directory)
        Tools.create_clear_directory(directory)

    # 2: load the content and style images
    if c_image is not None:
        content_image = c_image
    else:
        content_image = Tools.load_img(Config.content_path, Config.resize_input)
    if s_image is not None:
        style_image = s_image
    else:
        style_image = Tools.load_img(Config.style_path, Config.resize_input)

    # 3: plot the content and style images
    plt.subplot(1, 3, 1)
    Tools.imshow(content_image, 'Content Image')
    plt.subplot(1, 3, 2)
    Tools.imshow(style_image, 'Style Image')

    # 4: get fast style transfer output
    if Config.get_fast_style_output:
        plt.subplot(1, 3, 3)
        stylized_fast = get_stylized_image_from_fast_style_transfer(content_image, style_image)
        Tools.imshow(stylized_fast, 'Fast Style Transfer')
        Tools.tensor_to_image(stylized_fast).save(file_name + '_fast_stylized.png')

    # 5: run style transfer algorithm
    output_image, losses = do_style_transfer(content_image, style_image)

    # 6: pre-processing the output image
    output_image = Tools.tensor_to_image(output_image)
    if Config.show_log:
        output_image.save(file_name + '.png')
        plt.imshow(output_image)

        # 7: show losses
        plt.clf()
        x_axis = list(range(100, 100 + Config.epochs * 100, 100))
        plt.plot(x_axis, losses)
        plt.savefig(file_name + "_losses.png")
        plt.xlabel('iterations')
        plt.ylabel('total loss')
        plt.title('Image Style Transfer Total loss')
        plt.grid()
        plt.show()

    # 8: free memory
    global extractor
    global opt
    global style_targets
    global content_targets

    del extractor
    del opt
    del style_targets
    del content_targets

    return output_image
