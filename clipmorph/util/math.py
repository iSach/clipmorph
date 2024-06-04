def gram_matrix(y):
    """
    Computes the Gram matrix

    Arguments:
        y: Pytorch tensor of shape (batch size, channels, height, width).
    """
    (b, c, h, w) = y.size()
    features = y.view(b, c, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (c * h * w)
    return gram
