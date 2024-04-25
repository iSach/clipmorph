# test math util functions on randn data
import torch

from clipmorph.util.math import gram_matrix

def test_gram_matrix():

    y = torch.randn(1, 3, 256, 256)

    (b, c, h, w) = y.size()
    features = y.view(b, c, w * h)
    features_t = features.transpose(1, 2)
    gram_test = features.bmm(features_t) / (c * h * w)

    gram = gram_matrix(y)

    assert isinstance(gram, torch.Tensor)
    assert gram.size() == gram_test.size()
    assert torch.allclose(gram, gram_test)