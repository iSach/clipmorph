# Model Development

This page describes the development of our model, from choosing and 
collecting data, selecting the architecture, to training and evaluating the
model.

## Model Description

Our model is a so-called _"neural style transfer"_ model, i.e., a model 
based on deep learning that tries to replace the style of a video or an 
image by another style while keeping the content intact.

Our model is based on the pioneer work of Gatys et al. [^1], which 
introduced an optimization process for style transfer through the use of 
Gram matrices [^2] on the feature maps of a large pre-trained convolutional
neural network. This work was extended by Johnson et al. [^3] to avoid the 
per-image optimization process and to allow real-time style transfer. For 
this purpose, they train an image-to-image fully convolutional model that 
directly performs the entire optimization process in one forward pass. We 
complement this work by adding temporal consistency to the model, following 
the work of de Berker and Rainy [^4] that introduce a stabilization 
procedure through a specific addition of noise in the training process.

The resulting model is a U-Net-style [^5] architecture with 13 
convolutional layers (3 with down-sampling, 10 for processing, and 3 with
up-sampling).

> [!NOTE]
> This project is based upon work done in the "Deep Learning" course two 
> years earlier. More details about the original study of the model can be 
> found [here](https://github.com/iSach/video-nst). We have in this new 
> project improved the training loop, improved the architecture 
> hyperparameters, as well as accelerated the inference process.

## Data Preparation & EDA

## Training

## Evaluation

## Weights & Biases



### References
[^1]: Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). Image style transfer using convolutional neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2414-2423).
[^2]: https://en.wikipedia.org/wiki/Gram_matrix
[^3]: Johnson, J., Alahi, A., & Fei-Fei, L. (2016). Perceptual losses for 
real-time style transfer and super-resolution. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14 (pp. 694-711). Springer International Publishing.
[^4]: de Berker, A. and Rainy, J. (2018). [Stabilizing neural style-transfer 
for video](https://medium.com/element-ai-research-lab/stabilizing-neural-style-transfer-for-video-62675e203e42) on Medium.
[^5]: Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18 (pp. 234-241). Springer International Publishing.