import argparse

import numpy as np
import torch
import wandb
from torch import nn
from torch.optim import Adam
from torchvision import transforms as T
from tqdm import tqdm

from clipmorph.data import load_data, load_image
from clipmorph.nn import FastStyleNet, Vgg19
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
        use_wandb=True,
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

    data_loader = load_data(train_img_dir, batch_size, img_size=img_train_size)

    fsn = FastStyleNet().to(device)
    optimizer = Adam(fsn.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    vgg = Vgg19(device=device).to(device)

    fsn.compile()
    vgg.compile()

    transform = T.Compose([
        T.Resize(img_train_size),
        T.CenterCrop(img_train_size),
        T.ToTensor(),
        T.Lambda(lambda x: x.mul(255))
    ])
    style_img = load_image(style_img_path)
    style_img = transform(style_img)
    style_img = style_img.repeat(batch_size, 1, 1, 1).to(device)
    style_img = norm_batch_vgg(style_img)

    # Get style feature represenbtation
    feat_style = vgg(style_img)
    gram_style = [gram_matrix(i) for i in feat_style]


    for e in range(nb_epochs):
        fsn.train()
        count = 0
        step = 0

        for x in tqdm(data_loader):
            C, H, W = x.shape[1:]

            n_batch = len(x)
            count += n_batch
            x = x.to(device)
            optimizer.zero_grad()

            # noise_img = torch.randn_like(x) * 0.1
            noise_img = torch.randint_like(x, -noise, noise + 1, device=device)
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

            # Reconstruction (content): "relu3_3"
            L_content = content_weight * criterion(x_feat[2], y_feat[2])
            L_style = style_weight * style_loss(gram_style, y_feat, criterion,
                                                n_batch)
            L_tv = tv_weight * tot_variation_loss(y)

            # Small changes in the input should result in small changes in the output.
            L_temporal = temporal_weight * criterion(y, y_noisy)
            L_total = L_content + L_style + L_tv + L_temporal

            log_dict = {
                "total_loss": L_total.item(),
                "content_loss": L_content.item(),
                "style_loss": L_style.item(),
                "temporal_loss": L_temporal.item(),
                "tv_loss": L_tv.item()
            }

            if step % 50 == 0:
                np_img = y[0].permute(1, 2, 0).detach().cpu().numpy()
                in_np_img = x[0].permute(1, 2, 0).detach().cpu().numpy()
                np_img = np.concatenate((in_np_img, np_img), axis=1)

                log_dict["image"] = wandb.Image(np_img)

            if use_wandb:
                wandb.log(log_dict)

            L_total.backward()
            optimizer.step()
            step = step + 1

    # Save model
    save_model = name_model + ".pth"
    path_model = "./models/" + save_model
    torch.save(fsn.state_dict(), path_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a fast style network')
    parser.add_argument(
        '--train_img_dir', type=str,
        default="training_data/visual_genome/",
        help='Directory where the training images are stored (e.g., VisGenome)'
    )
    parser.add_argument(
        '--style_img_name', type=str,
        default="training_data/styles/starrynight.jpg",
        help='Path of the style image file'
    )
    parser.add_argument(
        '--img_train_size', type=int,
        default=512,
        help='Size of the training images (square)'
    )
    parser.add_argument(
        '--batch_size', type=int,
        default=8,
        help='Batch size for training'
    )
    parser.add_argument(
        '--nb_epochs', type=int,
        default=1,
        help='Number of epochs for training'
    )
    parser.add_argument(
        '--content_weight', type=float,
        default=1e5,
        help='Content loss weighting factor'
    )
    parser.add_argument(
        '--style_weight', type=float,
        default=4e10,
        help='Style loss weighting factor'
    )
    parser.add_argument(
        '--tv_weight', type=float,
        default=1e-6,
        help='Total variation loss weighting factor'
    )
    parser.add_argument(
        '--temporal_weight', type=float,
        default=1000,
        help='Temporal loss weighting factor'
    )
    parser.add_argument(
        '--noise_count', type=int,
        default=1000,
        help='Number of pixels to modify with noise'
    )
    parser.add_argument(
        '--noise', type=int,
        default=30,
        help='Range of noise to add'
    )
    parser.add_argument(
        '--name_model', type=str,
        default=None,
        help='Name of the model'
    )
    parser.add_argument(
        '--wandb',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--run-name", type=str,
        default="",
        help="Name of the run",
    )

    args = parser.parse_args()
    train_img_dir = args.train_img_dir
    style_img_name = args.style_img_name
    img_train_size = args.img_train_size
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    content_weight = args.content_weight
    style_weight = args.style_weight
    tv_weight = args.tv_weight
    temporal_weight = args.temporal_weight
    noise_count = args.noise_count
    noise = args.noise
    name_model = args.name_model
    if name_model is None:
        name_model = style_img_name.split("/")[-1].split(".")[0]
    use_wandb = args.wandb

    if use_wandb:
        run_name = args.run_name + f" ({name_model})"
        wandb.init(
            project="clipmorph",
            name=run_name,
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

    train(
        train_img_dir,
        img_train_size,
        style_img_name,
        batch_size,
        nb_epochs,
        content_weight,
        style_weight,
        tv_weight,
        temporal_weight,
        noise_count,
        noise,
        name_model,
        use_wandb=use_wandb
    )
