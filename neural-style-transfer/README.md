[1]: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
[2]: https://arxiv.org/pdf/1606.05897.pdf

# Original Neural Style Transfer

This is the original neural style transfer algorithm, introduced by Leon Gatys et al. in
[Image Style Transfer Using Convolutional Neural Networks][1]. The algorithm uses learned image representations
in deep convolutional neural networks to complete the style transfer.

While the original paper suggests the L-BFGS-B optimization method, this implementation uses Adam, since
L-BFGS-B is not included in TensorFlow.

Below is an example of the output of this method, where the style of
[The Great Wave off Kanagawa](images/wave.jpg) is transferred to the
[Astronaut Sloth](images/sloth.jpg).

![The Great Sloth off Kanagawa](images/stylized-sloth-wave.jpg)

The script `neural_style_transfer.py` contains an implementation of this algorithm and a command line interface
for performing the transfer on two images.

### Color Preservation

In [Preserving Color in Neural Artistic Style Transfer][2], Gatys et al. introduce an extension of the original
algorithm that allows for the color of the original image to be preserved during the style transfer process.
They offer two methods for doing this: color histogram matching and luminance-only transfer. The latter method
is implemented in `neural_style_transfer.py` as described in the paper, and can be enabled with a command
line flag.
