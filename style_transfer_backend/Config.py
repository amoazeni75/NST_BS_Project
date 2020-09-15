content_path = 'content_images/arvin2.jpg'
style_path = './style_images/picasso.jpg'

resize_input = True
use_avg_pool_layers = True
get_fast_style_output = False
show_log = False
max_dim = 512

# best parameters
total_variation_weight = 60
style_weight = 1e-2
content_weight = 1e4
epochs = 1
steps_per_epoch = 3
content_layers = ['block5_conv2']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

# more content
# content_weight = 1e8
# style_weight = 1e-5

# more style
# content_weight = 1e-2
# style_weight = 1e-5
