import random
import os

import wandb
from torch import nn
from torch.optim import Adam
from torchvision import transforms as T
from tqdm import tqdm

import numpy as np
import torch
import matplotlib.pyplot as plt

from clipmorph.data import load_data, load_image
from clipmorph.nn import FastStyleNet, Vgg19
from clipmorph.nn.backbone import norm_batch_vgg
from clipmorph.util import gram_matrix
from clipmorph.util.losses import style_loss, tot_variation_loss


def train(
        train_img_dir,
        img_train_size,
        style_img_path,
        batch_size,
        nb_epochs,
        content_weight,
        style_weight,
        tv_weight,
        temporal_weight,
        noise_count,
        noise,
        name_model,
        device='cuda'
):
    """
    Train a fast style network to generate an image with the style of the style image.

    Arguments:
        train_img_dir: Directory where the training images are stored (Genome)
        img_train_size: Size of the training images
        style_img_path: Name of the style image
        batch_size: Number of images per batch
        nb_epochs: Number of epochs
        content_weight: Weight of the content loss
        style_weight: Weight of the style loss
        tv_weight: Weight of the total variation loss
        temporal_weight: Weight of the temporal loss
        noise_count: Number of pixels to add noise to
        noise: Noise range
        name_model: Name of the model
        device: Device to use (default: 'cuda')

    Returns:
        The trained model.
    """

    data_loader = load_data(train_img_dir, img_train_size, batch_size)

    fsn = FastStyleNet().to(device)
    optimizer = Adam(fsn.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    vgg = Vgg19().to(device)

    transfo_style = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: x.mul(255))
    ])
    style_img = load_image(style_img_path)
    style_img = transfo_style(style_img)
    style_img = style_img.repeat(batch_size, 1, 1, 1).to(device)
    style_img = norm_batch_vgg(style_img)

    # Get style feature representation
    feat_style = vgg(style_img)
    gram_style = [gram_matrix(i) for i in feat_style]

    tot_loss = []
    styl_loss = []
    content_loss = []
    temp_loss = []
    tv_loss = []
    for e in range(nb_epochs):
        fsn.train()
        count = 0

        noise_img_a = np.zeros((3, img_train_size, img_train_size),
                               dtype=np.float32)
        # TODO in torch + sample.
        for _ in range(noise_count):
            x = random.randrange(img_train_size)
            y = random.randrange(img_train_size)
            noise_img_a[0][x][y] += random.randrange(-noise, noise)
            noise_img_a[1][x][y] += random.randrange(-noise, noise)
            noise_img_a[2][x][y] += random.randrange(-noise, noise)
        noise_img = torch.from_numpy(noise_img_a)
        noise_img = noise_img.to(device)

        for x in tqdm(data_loader):
            n_batch = len(x)
            count += n_batch
            x = x.to(device)
            optimizer.zero_grad()

            y_noisy = fsn(x + noise_img)
            y_noisy = norm_batch_vgg(y_noisy)

            y = fsn(x)
            x = norm_batch_vgg(x)
            y = norm_batch_vgg(y)
            x_feat = vgg(x)
            y_feat = vgg(y)

            # We take the output of layer "relu3_3" -> 2nd output of the list
            L_content = content_weight * criterion(x_feat[2], y_feat[2])
            L_style = style_weight * style_loss(gram_style, y_feat, criterion,
                                                n_batch)
            L_tv = tv_weight * tot_variation_loss(y)

            # Small changes in the input should result in small changes in the output.
            L_temporal = temporal_weight * criterion(y, y_noisy)
            L_total = L_content + L_style  # + L_tv + L_temporal
            tot_loss.append(L_total.item())
            content_loss.append(L_content.item())
            styl_loss.append(L_style.item())
            temp_loss.append(L_temporal.item())
            tv_loss.append(L_tv.item())

            wandb.log({
                "total_loss": L_total.item(),
                "content_loss": L_content.item(),
                "style_loss": L_style.item(),
                "temporal_loss": L_temporal.item(),
                "tv_loss": L_tv.item()
            })

            # if False:
            #    print('Epoch {}, L_total: {}, L_content {}, L_style {}, L_tot_var {}, L_temporal {}'
            #                    .format(e, L_total.data, L_content.data, L_style.data, L_tv.data, L_temporal.data))

            L_total.backward()
            optimizer.step()
            x.to('cpu')
        noise_img.to('cpu')  # release GPU memory

    # Save model
    fsn.eval().cpu()
    save_model = name_model + ".pth"
    path_model = "./models/" + save_model
    torch.save(fsn.state_dict(), path_model)

    # lease the GPU memory
    vgg.to('cpu')
    fsn.to('cpu')
    style_img.to('cpu')

    return tot_loss, content_loss, styl_loss, temp_loss, tv_loss


def plot_loss():
    train_img_dir = "training_data/visual_genome/"
    style_img_name = "training_data/styles/starrynight.jpg"
    img_train_size = 512
    batch_size = 8
    nb_epochs = 1
    content_weight = 1e5  # Content loss weighting factor
    style_weight = 4e10  # Style loss weighting factor
    tv_weight = 1e-6  # Total variation loss weighting factor
    temporal_weight = 1300  # Temporal loss weighting factor
    noise_count = 1000  # number of pixels to modify with noise
    noise = 30  # range of noise to add
    name_model = os.path.splitext(style_img_name)[0]  # name of the model
    # that will be saved after training

    wandb.init(
        project="clipmorph",
        name="debug",
        config={
            "train_img_dir": train_img_dir,
            "style_img_name": style_img_name,
            "img_train_size": img_train_size,
            "batch_size": batch_size,
            "nb_epochs": nb_epochs,
            "content_weight": content_weight,
            "style_weight": style_weight,
            "tv_weight": tv_weight,
            "temporal_weight": temporal_weight,
            "noise_count": noise_count,
            "noise": noise,
            "name_model": name_model
        }
    )

    tot_loss, content_loss, styl_loss, temp_loss, tv_loss = train(
        train_img_dir, img_train_size, style_img_name, batch_size, nb_epochs,
        content_weight, style_weight, tv_weight, temporal_weight, noise_count,
        noise, name_model)

    # Plot losses
    if False:
        plt.figure()
        plt.plot(tot_loss, label='Total loss')
        plt.legend()
        plt.figure()
        plt.plot(content_loss, label='Content loss')
        plt.legend()
        plt.figure()
        plt.plot(styl_loss, label='Style loss')
        plt.legend()
        plt.figure()
        plt.plot(temp_loss, label='Temporal loss')
        plt.legend()
        plt.figure()
        plt.plot(tv_loss, label='TV loss')
        plt.legend()
        plt.figure()
        plt.figure(figsize=(9, 6))
        plt.plot(tot_loss, label='Total loss')
        plt.plot(content_loss, label='Content loss')
        plt.plot(styl_loss, label='Style loss')
        plt.plot(temp_loss, label='Temporal loss')
        plt.plot(tv_loss, label='TV loss')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    plot_loss()