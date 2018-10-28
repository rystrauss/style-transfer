import sys
import time

import click
import numpy as np
from scipy.misc import imsave
from scipy.optimize import fmin_l_bfgs_b
from tensorflow.keras import backend as K
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array


@click.command()
@click.argument('target_path', type=click.Path())
@click.argument('reference_path', type=click.Path())
@click.option('--iterations', type=click.INT, default=20, help="The number of iterations to run the optimization.")
@click.option('--img_height', type=click.INT, default=400,
              help='The height of the output image. Width will be based on the aspect ratio of the target image.')
@click.option('--tv_weight', type=click.FLOAT, default=0.0001, help='The weight given to the total variation loss.')
@click.option('--style_weight', type=click.FLOAT, default=1., help='The weight given to the style loss.')
@click.option('--content_weight', type=click.FLOAT, default=0.025, help='The weight given to the content loss.')
@click.option('--save_every', type=click.INT, default=sys.maxsize,
              help='The frequency with with to save the output image.')
@click.option('--verbose', type=click.BOOL, default=False, help='Specifies the output verbosity.')
def main(target_path, reference_path, iterations, img_height, tv_weight,
         style_weight, content_weight, save_every, verbose):
    """Performs neural style transfer on a target image and reference image.

    The style of the reference image is imposed onto the target image. This is the original implementation of
    neural style transfer proposed by Leon Gatys et al. 2015. It is preferable to run this script on GPU, for speed.

    The code in this script is adapted from Francois Chollet's 'Deep Learning with Python'.
    """

    def preprocess_image(image_path):
        """Loads and preprocesses the specified image."""
        img = load_img(image_path, target_size=(img_height, img_width))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = vgg19.preprocess_input(img)
        return img

    def deprocess_image(x):
        """Zero-centering by removing the mean pixel value from ImageNet.
        This reverses a transformation done by vgg19.preprocess_input.
        """
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        x = x[:, :, ::-1]
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    def content_loss(base, combination):
        """Defines the content loss function."""
        return K.sum(K.square(combination - base))

    def gram_matrix(x):
        """Returns the Gram matrix of the input."""
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
        gram = K.dot(features, K.transpose(features))
        return gram

    def style_loss(style, combination):
        """Defines the style loss function."""
        S = gram_matrix(style)
        C = gram_matrix(combination)
        channels = 3
        size = img_height * img_width
        return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

    def total_variation_loss(x):
        """Defines the total variation loss function.
        This can be interpreted as a regularization loss.
        """
        a = K.square(
            x[:, :img_height - 1, :img_width - 1, :] -
            x[:, 1:, :img_width - 1, :])
        b = K.square(
            x[:, :img_height - 1, :img_width - 1, :] -
            x[:, :img_height - 1, 1:, :])
        return K.sum(K.pow(a + b, 1.25))

    class Evaluator(object):
        """This class wraps fetch_loss_and_grads in a way that allows the retrieval
        of the losses and gradients via two separate method calls, which is required
        by the SciPy optimizer that is used."""

        def __init__(self):
            self.loss_value = None
            self.grads_values = None

        def loss(self, x):
            assert self.loss_value is None
            x = x.reshape((1, img_height, img_width, 3))
            outs = fetch_loss_and_grads([x])
            loss_value = outs[0]
            grad_values = outs[1].flatten().astype('float64')
            self.loss_value = loss_value
            self.grad_values = grad_values
            return self.loss_value

        def grads(self, x):
            assert self.loss_value is not None
            grad_values = np.copy(self.grad_values)
            self.loss_value = None
            self.grad_values = None
            return grad_values

    # Set image dimensions
    width, height = load_img(target_path).size
    img_width = int(width * img_height / height)

    # Define images
    target_image = K.constant(preprocess_image(target_path))
    reference_image = K.constant(preprocess_image(reference_path))
    combination_image = K.placeholder((1, img_height, img_width, 3))

    # Concatenate images for batch processing
    input_tensor = K.concatenate([target_image,
                                  reference_image,
                                  combination_image], axis=0)

    # Load in the VGG19 network
    model = vgg19.VGG19(input_tensor=input_tensor,
                        weights='imagenet',
                        include_top=False)

    # Build a layer dictionary and define the layers to be used for content and style
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    content_layer = 'block5_conv2'
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    # Variable that holds the loss. All loss components will be added to this variable
    loss = K.variable(0.)

    # Add the content loss
    layer_features = outputs_dict[content_layer]
    target_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight * content_loss(target_image_features,
                                                combination_features)

    # Add the style loss
    for layer_name in style_layers:
        layer_features = outputs_dict[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss = loss + (style_weight / len(style_layers)) * sl

    # Add the total variation loss
    loss = loss + tv_weight * total_variation_loss(combination_image)

    # Get the gradients of the generated image with regard to the loss
    grads = K.gradients(loss, combination_image)[0]

    # Function to fetch the values of the current loss and the current gradients
    fetch_loss_and_grads = K.function([combination_image], [loss, grads])

    evaluator = Evaluator()

    # The target image is the initial state
    x = preprocess_image(target_path)
    x = x.flatten()

    # Run L-BFGS optimization over the pixels of the generated image to minimize the neural style loss.
    for i in range(iterations):
        if verbose:
            print('Start of iteration', i + 1)
            start_time = time.time()

        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads, maxfun=20)

        if verbose:
            print('Current loss value:', min_val)

        if i == iterations - 1 or (i + 1) % save_every == 0:
            img = x.copy().reshape((img_height, img_width, 3))
            img = deprocess_image(img)
            fname = '{}-iter-{}.png'.format(target_path, i + 1)
            imsave(fname, img)
            if verbose:
                print('Image saved.')

        if verbose:
            end_time = time.time()
            print('Iteration %d completed in %ds' % (i + 1, end_time - start_time))
            print()


if __name__ == '__main__':
    main()
