"""Implementation of neural style transfer.

The style of the reference image is imposed onto the target image.
This is the original implementation of neural style transfer proposed by
Leon Gatys et al. 2015. It is preferable to run this script on GPU, for speed.

Parts of this implementation are adapted from Google's Colab example found here:
https://colab.research.google.com/github/tensorflow/models/blob/master/research/nst_blogpost/4_Neural_Style_Transfer_with_Eager_Execution.ipynb

Author: Ryan Strauss
"""
import time

import click
import imageio
import numpy as np
import tensorflow as tf
from skimage.color import rgb2yiq, yiq2rgb
from skimage.io import imread, imsave
from skimage.transform import resize

CONTENT_LAYERS = ['block5_conv2']
STYLE_LAYERS = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

NUM_CONTENT_LAYERS = len(CONTENT_LAYERS)
NUM_STYLE_LAYERS = len(STYLE_LAYERS)

NORM_MEANS = np.array([103.939, 116.779, 123.68])
MIN_VALS = -NORM_MEANS
MAX_VALS = 255 - NORM_MEANS


def load_img(image_path, image_height, as_gray=False):
    """Loads an image.

    Args:
        image_path: Path for the image to be loaded.
        image_height: The height the image should be resized to.
        as_gray: If true, image will be read in as grayscale.

    Returns:
        The loaded image.
    """
    # Read in the image
    img = imread(image_path, as_gray=as_gray)
    # If it's grayscale, give it three channels
    if as_gray:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)
        img = (img * 255).astype('uint8')
    if image_height:
        # Calculate image width based off aspect ratio of image
        img_width = img.shape[1] * image_height // img.shape[0]
        # Resize the image
        img = resize(
            img, (image_height, img_width), anti_aliasing=True, mode='reflect',
            preserve_range=True).astype('uint8')
    return img


def load_and_process_img(image_path, image_height, as_gray=False):
    """Loads and preprocesses an image.

    Image is resized and proprocessed for the VGG19 network.

    Args:
        image_path: Path for the image to be loaded.
        image_height: The height the image should be resized to.
        as_gray: If true, image will be read in as grayscale.

    Returns:
        The processed image.
    """
    img = load_img(image_path, image_height, as_gray)
    # Give image a batch dimension
    img = np.expand_dims(img, axis=0)
    # Preprocess for VGG19 network
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img.astype('float32')


def deprocess_image(processed_img):
    """Undo preprocessing that was applied to an image.

    Args:
        processed_img: An image that has been preprocessed.

    Returns:
        An image in standard uint8 form.
    """
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)

    # Perform the inverse of the VGG19 preprocessing
    x[:, :, 0] += NORM_MEANS[0]
    x[:, :, 1] += NORM_MEANS[1]
    x[:, :, 2] += NORM_MEANS[2]
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x


def content_loss(base_content, target):
    """Calculates the content loss between the base features and the
    target features.

    This is simply the squared error between the layer responses for two images.

    Args:
        base_content: The base content.
        target: The target.

    Returns:
        The content loss between the two features.
    """
    return tf.reduce_mean(tf.square(base_content - target))


def gram_matrix(input_tensor):
    """Computes the Gram matrix of the input.

    Args:
        input_tensor: A Tensor for which to calculate the Gram matrix.

    Returns:
        The Gram matrix of the input tensor.
    """
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def style_loss(base_style, gram_target):
    """Calculates the style loss between the base content and the target.

    Args:
        base_style: The base image's style features.
        gram_target: The Gram matrix of the target features.

    Returns:
        The style loss.
    """
    gram_style = gram_matrix(base_style)

    return tf.reduce_mean(tf.square(gram_style - gram_target))


def get_model():
    """Creates a VGG19 model with access to intermediate layers.

    This function will load the VGG19 model and access the intermediate layers.
    These layers will then be used to create a new model that will take an
    input image and return the outputs from the intermediate layers of the
    VGG model.

    Returns:
        A model that takes image inputs and outputs the style and content
        intermediate layers.
    """
    # Load the pretrained VGG19 network
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False,
                                            weights='imagenet')
    vgg.trainable = False

    # Get output layers corresponding to style and content layers
    style_outputs = [vgg.get_layer(name).output for name in STYLE_LAYERS]
    content_outputs = [vgg.get_layer(name).output for name in CONTENT_LAYERS]
    model_outputs = style_outputs + content_outputs

    # Build model
    model = tf.keras.Model(vgg.input, model_outputs)

    for layer in model.layers:
        layer.trainable = False
    return model


def tv_loss(input_tensor):
    """Computes the total variation loss for a tensor.

    This loss encourages spatial continuity in the generated image, thus
    avoiding overly pixelated results. While Gatys' original algorithm does
    not use this loss, it has been shown to improve results on the images
    stylization task by reducing the amount of high frequency noise in the
    output image.

    Args:
        input_tensor: A Tensor for which to compute the TV loss.

    Returns:
        The TV loss for the input.
    """
    a = tf.square(input_tensor[:, :-1, :-1, :] - input_tensor[:, 1:, :-1, :])
    b = tf.square(input_tensor[:, :-1, :-1, :] - input_tensor[:, :-1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, 1.25))


def compute_loss(model, loss_weights, init_image, gram_style_features,
                 content_features):
    """This function will compute the loss total loss.

    Arguments:
        model: The model that will give us access to the intermediate layers
        loss_weights: The weights of each contribution of each loss function.
            (style weight, content weight, and total variation weight)
        init_image: Our initial base image. This image is what we are updating
            with our optimization process. We apply the gradients wrt the
            loss we are calculating to this image.
        gram_style_features: Precomputed gram matrices corresponding to the
            defined style layers of interest.
        content_features: Precomputed outputs from defined content layers of
            interest.

    Returns:
        The total loss.
    """
    style_weight, content_weight, tv_weight = loss_weights

    # Feed the init image through the network
    model_outputs = model(init_image)

    style_output_features = model_outputs[:NUM_STYLE_LAYERS]
    content_output_features = model_outputs[NUM_STYLE_LAYERS:]

    style_score = 0
    content_score = 0

    # Accumulate style losses from all layers
    # We equally weight each contribution of each loss layer
    for target_style, comb_style in zip(gram_style_features,
                                        style_output_features):
        style_score += style_loss(comb_style[0], target_style)
    style_score /= float(NUM_STYLE_LAYERS)

    # Accumulate content losses from all layers
    for target_content, comb_content in zip(content_features,
                                            content_output_features):
        content_score += content_loss(comb_content[0], target_content)
    content_score /= float(NUM_CONTENT_LAYERS)

    style_score *= style_weight
    content_score *= content_weight
    tv_score = tv_weight * tv_loss(init_image)

    return style_score + content_score + tv_score


@tf.function
def compute_grads(config):
    """Compute gradients and loss.

    Args:
        config: A dictionary containing the following:
            config = {
                'model': model,
                'loss_weights': loss_weights,
                'init_image': init_image,
                'gram_style_features': gram_style_features,
                'content_features': content_features
            }

    Returns:
        The tuple (gradients, loss).
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(**config)
    # Compute gradients wrt input image
    return tape.gradient(loss, config['init_image']), loss


def create_gif(images, path):
    """Create a GIF from the provided images.

    Args:
        images: A list of images as arrays.
        path: Where the GIF should be saved.

    Returns:
        None
    """
    with imageio.get_writer(path, mode='I') as writer:
        for image in images:
            writer.append_data(image)


@click.command()
@click.argument('content_image_path',
                type=click.Path(file_okay=True, dir_okay=False, exists=True),
                nargs=1)
@click.argument('style_image_path',
                type=click.Path(file_okay=True, dir_okay=False, exists=True),
                nargs=1)
@click.option('--iterations', type=click.INT, default=100, nargs=1,
              help='The number of iterations to run the optimization. '
                   'Default is 100.')
@click.option('--content_img_height', type=click.INT, default=None, nargs=1,
              help='The height of the output image. Width will be based on the '
                   'aspect ratio of the target image.')
@click.option('--style_img_height', type=click.INT, default=None, nargs=1,
              help='The height of the reference image. Width will be based on '
                   'the aspect ratio of the target image.')
@click.option('--tv_weight', type=click.FLOAT, default=0.001, nargs=1,
              help='The weight given to the total variation loss. '
                   'Default is 0.001.')
@click.option('--style_weight', type=click.FLOAT, default=1., nargs=1,
              help='The weight given to the style loss. Default is 1.')
@click.option('--content_weight', type=click.FLOAT, default=0.1, nargs=1,
              help='The weight given to the content loss. A higher value means '
                   'the target image content will be more recognizable in the '
                   'output. Default is 0.1.')
@click.option('--save_gif', is_flag=True,
              help='If flag is set, a GIF will be saved showing the '
                   'stylization process.')
@click.option('--preserve_color', is_flag=True,
              help='Enables color preservation.')
@click.option('--learning_rate', type=click.FLOAT, default=5., nargs=1,
              help='Adam learning rate. Default is 5.')
@click.option('--beta_1', type=click.FLOAT, default=0.99, nargs=1,
              help='Value of beta1 in Adam optimizer. Default is 0.99.')
@click.option('--beta_2', type=click.FLOAT, default=0.999, nargs=1,
              help='Value of beta2 in Adam optimizer. Default is 0.999.')
@click.option('--epsilon', type=click.FLOAT, default=0.1, nargs=1,
              help='Value of epsilon in Adam optimizer. Default is 0.1.')
def main(content_image_path, style_image_path, iterations, content_img_height,
         style_img_height, tv_weight,
         style_weight, content_weight, save_gif, preserve_color, learning_rate,
         beta_1, beta_2, epsilon):
    """Performs neural style transfer on a content image and style image."""
    model = get_model()

    # Load images
    content_image = load_and_process_img(content_image_path, content_img_height)
    style_image = load_and_process_img(style_image_path, style_img_height)
    content_image_yiq = rgb2yiq(
        load_img(content_image_path, content_img_height))

    # Compute content and style features
    style_outputs = model(style_image)
    content_outputs = model(content_image)

    # Get the style and content feature representations from our model
    style_features = [style_layer[0] for style_layer in
                      style_outputs[:NUM_STYLE_LAYERS]]
    content_features = [content_layer[0] for content_layer in
                        content_outputs[NUM_STYLE_LAYERS:]]
    gram_style_features = [gram_matrix(style_feature) for style_feature in
                           style_features]

    # Set initial image
    init_image = load_and_process_img(content_image_path, content_img_height,
                                      as_gray=preserve_color)
    init_image = tf.Variable(init_image, dtype=tf.float32)

    # Create our optimizer
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1,
                                   beta_2=beta_2, epsilon=epsilon)

    # Create config dictionary
    loss_weights = (style_weight, content_weight, tv_weight)
    config = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }

    images = []

    # Optimization loop
    for step in range(1, iterations + 1):
        start_time = time.time()
        grads, loss = compute_grads(config)
        optimizer.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, MIN_VALS, MAX_VALS)
        init_image.assign(clipped)
        img = deprocess_image(init_image.numpy())
        if preserve_color:
            img = rgb2yiq(img)
            img[:, :, 1:] = content_image_yiq[:, :, 1:]
            img = yiq2rgb(img)
            img = np.clip(img, 0, 1)
            img = (img * 255).astype('uint8')
        images.append(img)
        end_time = time.time()
        print('Finished step {} ({:.03} seconds)\nLoss: {}\n'.format(
            step, end_time - start_time, loss))

    # Save final image
    imsave('stylized.jpg', images[-1])

    if save_gif:
        create_gif(images, 'transformation.gif')


if __name__ == '__main__':
    main()
