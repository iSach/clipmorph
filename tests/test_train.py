# Test training on dummy dataset of 60 images.
import numpy as np
import torch

from torch import nn
from clipmorph.data import load_data
from clipmorph.data.genome_loader import load_image
from clipmorph.nn.backbone import Vgg19
from clipmorph.nn.unet import FastStyleNet
from torch.optim import Adam
from torchvision import transforms as T
from clipmorph.util.losses import style_loss, tot_variation_loss
from clipmorph.util.math import gram_matrix

def train_test():
    """ Test training on dummy dataset of 5 images."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = load_data('./training_data/styles', 7, img_size=224)
    fsn = FastStyleNet().to(device)
    vgg = Vgg19(device=device).to(device)
    fsn_params = sum(p.numel() for p in fsn.parameters() if p.requires_grad)

    if torch.cuda.is_available():
        fsn.compile()
        vgg.compile()

    optimizer = Adam(fsn.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    transform = T.Compose(
        [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Lambda(lambda x: x.mul(255)),
        ]
    )
    style_img = load_image('./training_Data/styles/turner.jpg')
    style_img = transform(style_img)

    style_img = style_img.repeat(7, 1, 1, 1).to(device)
    style_img = vgg.normalize_batch(style_img)

    # Get style feature represenbtation
    feat_style = vgg(style_img)
    gram_style = [gram_matrix(i) for i in feat_style]

    fsn.train()

    data_iter = iter(data_loader)

    for step in range(5):
        try:
            x = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            x = next(data_iter)

        n_batch = len(x)
        x = x.to(device)
        optimizer.zero_grad()

        # noise_img = torch.randn_like(x) * 0.1
        noise_img = torch.randint_like(x, -30, 30 + 1, device=device)
        mask = torch.rand_like(x, device=device) < 0.05
        noise_img = noise_img * mask.float()

        y_noisy = fsn(x + noise_img)
        y_noisy = vgg.normalize_batch(y_noisy)

        y = fsn(x)
        x = vgg.normalize_batch(x)
        y = vgg.normalize_batch(y)
        x_feat = vgg(x)
        y_feat = vgg(y)
        # Features at relu1_2, relu2_2, relu3_3, relu4_3

        # Reconstruction (content): "relu2_2"
        L_content = 1e5 * criterion(x_feat[1], y_feat[1])
        L_style = 4e10 * style_loss(gram_style, y_feat, criterion, n_batch)
        L_tv = 1e-6 * tot_variation_loss(y)

        # Small changes in the input should result in small changes in the output.
        L_temporal = 1000 * criterion(y, y_noisy)
        L_total = L_content + L_style + L_tv + L_temporal

        L_total.backward()
        torch.nn.utils.clip_grad_norm_(fsn.parameters(), 4.0)
        optimizer.step()
