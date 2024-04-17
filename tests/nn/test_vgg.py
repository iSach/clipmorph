import torch

from clipmorph.nn.backbone import Vgg19

def test_vgg_load():

    vgg = Vgg19(device='cpu')
    assert vgg is not None

    X = torch.randn(4, 3, 224, 224)

    Y = vgg.normalize_batch(X)

    assert Y is not None
    assert X.shape == Y.shape

    vgg_outputs = vgg(Y)
    assert len(vgg_outputs) == 4