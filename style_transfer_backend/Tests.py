from PIL import Image
from ContentReconstruction import test_content_reconstruction
from StyleReconstruction import test_style_reconstruction
from Tools import resize_image, create_clear_directory, clean_gpu_memory, load_img_and_preprocess_resize, load_img
import StyleTransfer2
import Config
import gc


# 2
def compare_vgg19_blocks_for_content_layer_reconstruction():
    mydir = './Outputs/compare_vgg19_blocks_for_content_layer_reconstruction'
    create_clear_directory(mydir)

    images = []
    layers = [1, 4, 9, 14, 19]

    # select content images
    images.append(resize_image(Image.open('content_images/tubingen.jpg'), 512))
    images.append(resize_image(Image.open('content_images/old-city-jail.jpg'), 512))
    images.append(resize_image(Image.open('content_images/hoovertowernight.jpg'), 512))
    images.append(resize_image(Image.open('content_images/golden_gate.jpg'), 512))
    images.append(resize_image(Image.open('content_images/5.jpg'), 512))

    counter = 0
    for img in images:
        counter += 1
        img.save(mydir + "/original_{number}.jpg".format(number=counter))
        test_content_reconstruction(img, mydir, layers, counter)
    gc.collect()


# 3
def compare_vgg19_5ht_block_layer():
    mydir = './Outputs/compare_vgg19_5ht_block_layer_content_reconstruction'
    create_clear_directory(mydir)

    images = []
    layers = [13, 14, 16, 17]

    # select content images
    # images.append(resize_image(Image.open('content_images/tubingen.jpg'), 512))
    # images.append(resize_image(Image.open('content_images/old-city-jail.jpg'), 512))
    # images.append(resize_image(Image.open('content_images/hoovertowernight.jpg'), 512))
    images.append(resize_image(Image.open('content_images/golden_gate.jpg'), 512))
    images.append(resize_image(Image.open('content_images/5.jpg'), 512))

    counter = 3
    for img in images:
        counter += 1
        img.save(mydir + "/original_{number}.jpg".format(number=counter))
        test_content_reconstruction(img, mydir, layers, counter)
        gc.collect()


# 4
def compare_different_number_layer_style_reconstruction():
    mydir = './Outputs/compare_different_number_layer_style_reconstruction'
    create_clear_directory(mydir)

    images = []
    # select content images
    images.append(resize_image(Image.open('style_images/starry_night.jpg'), 512))
    images.append(resize_image(Image.open('style_images/picasso.jpg'), 512))
    images.append(resize_image(Image.open('style_images/4.jpg'), 512))
    images.append(resize_image(Image.open('style_images/frida_kahlo.jpg'), 512))
    images.append(resize_image(Image.open('style_images/1.jpg'), 512))

    counter = 0
    for img in images:
        counter += 1
        img.save(mydir + "/original_{number}.jpg".format(number=counter))
        test_style_reconstruction(img,
                                  mydir,
                                  counter)
    clean_gpu_memory()


def main():
    # 2
    # compare_vgg19_blocks_for_content_layer_reconstruction()

    # 3
    # compare_vgg19_5ht_block_layer()

    # 4
    # compare_different_number_layer_style_reconstruction()

    # style_transfer('content_images/tubingen.jpg', 'style_images/starry_night.jpg')
    content_image = load_img(Config.content_path, Config.resize_input)
    style_image = load_img(Config.style_path, Config.resize_input)
    StyleTransfer2.style_transfer(content_image, style_image)
    print("hello")


if __name__ == "__main__":
    main()
