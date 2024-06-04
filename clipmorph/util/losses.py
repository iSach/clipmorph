import torch

from clipmorph.util import gram_matrix


def style_loss(gram_style_img, curr_img_features, criterion, n_batch):
    """
    Sum MSE's from the chosen output layers of VGG

    Arguments:
        gram_style_img: Gram matrix of the style representation of the style image
        curr_img_features: Output of the VGG network for the current image
        criterion: Loss criterion
        n_batch: Number of images in the batch
    """

    L_style = 0
    for curr_feat, gram_style in zip(curr_img_features, gram_style_img):
        curr_gram = gram_matrix(curr_feat)
        L_style += criterion(curr_gram, gram_style[:n_batch, :, :])
    return L_style


def tot_variation_loss(img):
    """
    Computes the total variation loss.

    From: https://en.wikipedia.org/wiki/Total_variation_denoising
    """

    loss = torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + torch.sum(
        torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
    )
    return loss
