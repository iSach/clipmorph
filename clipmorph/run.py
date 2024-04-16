import os

import torch
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from clipmorph.data import load_image
from clipmorph.nn import FastStyleNet


def neural_style_transfer(model_path, content_dir, output_dir):
    """
    Apply a model to all images in a directory and save the result in another directory.

    Args:
        model_path: path to the model
        content_dir: path to the directory containing the content images
        output_dir: path to the directory where the output images will be saved
    """

    style_model = FastStyleNet()
    style_model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    style_model.to(device)

    img_list = os.listdir(content_dir)

    for i in tqdm(range(len(img_list))):
        img_path = content_dir + str(i) + ".jpg"
        content_image = load_image(img_path)
        content_trans = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: x.mul(255))
        ])
        content_img = content_trans(content_image)

        if content_img.size()[0] != 3:
            content_img = content_img.expand(3, -1, -1)
        content_img = content_img.unsqueeze(0).to(device)

        with torch.no_grad():
            stylized = style_model(content_img).cpu()
        stylized = stylized[0]
        stylized = stylized.clone().clamp(0, 255).numpy()
        stylized = stylized.transpose(1, 2, 0).astype("uint8")
        stylized = Image.fromarray(stylized)
        stylized.save(output_dir + str(i) + ".jpg")
