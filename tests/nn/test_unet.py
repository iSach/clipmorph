import torch

from clipmorph.nn.unet import FastStyleNet

def test_faststylenet():
    """Test FastStyleNet creation & forward pass."""

    net = FastStyleNet()
    assert isinstance(net, torch.nn.Module)

    x = torch.randn(1, 3, 256, 256)
    assert x.requires_grad == False
    y = net(x)
    assert y.size() == x.size()