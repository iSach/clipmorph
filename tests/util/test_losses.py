# test losses on randn data
import torch
from clipmorph.util.losses import tot_variation_loss

def test_loss():
    img = torch.randn(1, 3, 224, 224)

    tot_loss = torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + torch.sum(
            torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
        )
    assert tot_variation_loss(img) == tot_loss

