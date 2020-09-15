from PIL import Image
from style_transfer_backend import ContentReconstruction, StyleReconstruction, \
    Tools, StyleTransfer2, Config
import gc


# 2
def compare_vgg19_blocks_for_content_layer_reconstruction():
    mydir = '../media/results/compare_vgg19_blocks_for_content_layer_reconstruction'
    Tools.create_clear_directory(mydir)

    images = []
    layers = [1, 4, 9, 14, 19]

    # select content images
    images.append(Tools.resize_image(Image.open('../media/contentfile/tubingen.jpg'), 512))
    images.append(Tools.resize_image(Image.open('../media/contentfile/old-city-jail.jpg'), 512))
    images.append(Tools.resize_image(Image.open('../media/contentfile/hoovertowernight.jpg'), 512))
    images.append(Tools.resize_image(Image.open('../media/contentfile/golden_gate.jpg'), 512))
    images.append(Tools.resize_image(Image.open('../media/contentfile/5.jpg'), 512))

    counter = 0
    for img in images:
        counter += 1
        img.save(mydir + "/original_{number}.jpg".format(number=counter))
        ContentReconstruction.test_content_reconstruction(img, mydir, layers, counter)
    gc.collect()


# 3
def compare_vgg19_5ht_block_layer():
    mydir = '../media/results/compare_vgg19_5ht_block_layer_content_reconstruction'
    Tools.create_clear_directory(mydir)

    images = []
    layers = [13, 14, 16, 17]

    # select content images
    images.append(Tools.resize_image(Image.open('../media/contentfile/tubingen.jpg'), 512))
    images.append(Tools.resize_image(Image.open('../media/contentfile/old-city-jail.jpg'), 512))
    images.append(Tools.resize_image(Image.open('../media/contentfile/hoovertowernight.jpg'), 512))
    images.append(Tools.resize_image(Image.open('../media/contentfile/golden_gate.jpg'), 512))
    images.append(Tools.resize_image(Image.open('../media/contentfile/5.jpg'), 512))

    counter = 0
    for img in images:
        counter += 1
        img.save(mydir + "/original_{number}.jpg".format(number=counter))
        ContentReconstruction.test_content_reconstruction(img, mydir, layers, counter)
        gc.collect()


# 4
def compare_different_number_layer_style_reconstruction():
    mydir = '../media/results/compare_different_number_layer_style_reconstruction'
    Tools.create_clear_directory(mydir)

    images = []
    # select content images
    images.append(Tools.resize_image(Image.open('../media/stylefile/starry_night.jpg'), 512))
    images.append(Tools.resize_image(Image.open('../media/stylefile/picasso.jpg'), 512))
    images.append(Tools.resize_image(Image.open('../media/stylefile/4.jpg'), 512))
    images.append(Tools.resize_image(Image.open('../media/stylefile/frida_kahlo.jpg'), 512))
    images.append(Tools.resize_image(Image.open('../media/stylefile/1.jpg'), 512))

    counter = 0
    for img in images:
        counter += 1
        img.save(mydir + "/original_{number}.jpg".format(number=counter))
        StyleReconstruction.test_style_reconstruction(img,
                                                      mydir,
                                                      counter)
    gc.collect()


def main():
    # 2
    # compare_vgg19_blocks_for_content_layer_reconstruction()

    # 3
    # compare_vgg19_5ht_block_layer()

    # 4
    # compare_different_number_layer_style_reconstruction()

    # style_transfer('content_images/tubingen.jpg', 'style_images/starry_night.jpg')
    content_image = Tools.load_img(Config.content_path, Config.resize_input)
    style_image = Tools.load_img(Config.style_path, Config.resize_input)
    StyleTransfer2.style_transfer(content_image, style_image)


if __name__ == "__main__":
    main()
