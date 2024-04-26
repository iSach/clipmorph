# test losses on randn data
# test losses on randn data
import torch
from clipmorph.util.losses import tot_variation_loss, style_loss
from clipmorph.nn.backbone import Vgg19

img = torch.randn(1, 3, 224, 224)

tot_loss = torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + torch.sum(
        torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
    )
assert tot_variation_loss(img) == tot_loss


def test_vgg_load():
    vgg = Vgg19(device="cpu")
    X = torch.randn(4, 3, 224, 224)
    Y = vgg.normalize_batch(X)
    vgg_outputs = vgg(Y)
