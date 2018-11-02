[1]: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
[2]: https://arxiv.org/pdf/1606.05897.pdf


# Neural Style Transfer
Neural style transfer is a collection of techniques that come out of deep learning that allow the style of one image
to be transfered to the content of another. The method was first introduced by [Gatys et al.][1] in 2016.

This repository contains implementations of the following style transfer algorithms in TensorFlow's Keras API.

## Prerequisites

The code in this repository relies on the following python packages, all of which can be installed via `pip`:

* `tensorflow`

   *Using `tensorflow-gpu` is recommended if possible, as GPUs will dramatically decrease runtime.*
   
* `numpy`

* `scipy`

* `click` (for command-line interface)

## Original Neural Style Transfer
This is the original neural style transfer algorithm, introduced by Leon Gatys et al. in
[Image Style Transfer Using Convolutional Neural Networks][1]. The algorithm uses learned image representations
in deep convolutional neural networks to complete the style transfer.

Below is an example of the output of this method, where the style of The Great Wave off Kanagawa is transferred
to the Astronaut Sloth.

[Astronaut Sloth](examples/target-sloth.jpg)
[The Great Wave off Kanagawa](examples/reference-wave.jpg)
[The Great Sloth off Kanagawa](examples/stylized-sloth-wave.jpg)

The script `neural_style_transfer.py` contains an implementation of this algorithm and a command line interface
for performing the transfer on two images. The code is adapted from Francois Chollet's *Deep Learning with Python*.

In [Preserving Color in Neural Artistic Style Transfer][2], Gatys et al. introduce an extension of the original
algorithm that allows for the color of the original image to be preserved during the style transfer process.
They offer two methods for doing this: color histogram matching and luminance-only transfer. The latter method
is implemented in `neural_style_transfer.py` as described in the paper, and can be enabled or disabled with a command
line flag.
