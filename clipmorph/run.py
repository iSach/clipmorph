import argparse
import os

import torch
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from clipmorph.data import load_image, Data, load_data
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
    style_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    style_model.to(device)

    data = load_data(content_dir, 16)

    stylized_imgs = []
    for i, img in tqdm(enumerate(data)):
        img = img.to(device)

        with torch.no_grad():
            stylized = style_model(img).cpu()

        stylized_imgs.append(stylized)

    stylized_imgs = torch.cat(stylized_imgs, dim=0)
    print(stylized_imgs.shape)

    """
    stylized = stylized[0]
    stylized = stylized.clone().clamp(0, 255).numpy()
    stylized = stylized.transpose(1, 2, 0).astype("uint8")
    stylized = Image.fromarray(stylized)
    stylized.save(output_dir + str(i) + ".jpg")
    """



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a pre-trained model.')
    parser.add_argument(
        '--model', type=str,
        default="models/gericault.pth",
        help='Path to the model .pth file',
    )
    parser.add_argument(
        '--source', type=str,
        help='Path of the image/video to stylize',
        required=True,
    )
    parser.add_argument(
        '--output', type=str,
        default=None,
        help='Path of the output image/video',
    )

    args = parser.parse_args()
    if args.output is None:
        args.output = os.path.splitext(args.source)[0] + "_stylized" + os.path.splitext(args.source)[1]
    neural_style_transfer(args.model, args.source, args.output)