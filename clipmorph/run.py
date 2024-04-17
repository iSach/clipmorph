import argparse
import os

import torch
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from clipmorph.data import load_data
from clipmorph.nn import FastStyleNet


def stylize_video(model, video_path, output_path):
    video_name = video_path.split('/')[-1].split('.')[0]

    # Create temporary folder for frames
    temp_folder_name = f'temp_{video_name}'
    os.makedirs(temp_folder_name, exist_ok=True)

    # Extract frames from video
    os.system(f'ffmpeg -hide_banner -loglevel error -i {video_path} -q:v 2'
              f' {temp_folder_name}/frame_%d.jpg')

    style_model = FastStyleNet()
    style_model.load_state_dict(torch.load(model))
    style_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    style_model.to(device)

    data = load_data(temp_folder_name, 16)
    num_images = data.dataset.num_images

    progress_bar = tqdm(total=num_images, desc="Stylizing images")

    stylized_imgs = []
    stylized_img_names = []
    for img, names in data:
        img = img.to(device)

        with torch.no_grad():
            stylized = style_model(img).cpu()

        stylized_imgs.append(stylized)
        stylized_img_names.extend(names)

        progress_bar.update(len(names))

    stylized_imgs = torch.cat(stylized_imgs, dim=0)

    for i, stylized in enumerate(stylized_imgs):
        stylized = stylized.clone().clamp(0, 255).numpy()
        stylized = stylized.transpose(1, 2, 0).astype("uint8")
        stylized = Image.fromarray(stylized)
        stylized_path = temp_folder_name + "/" + stylized_img_names[i].split(
            "/")[
            -1].split(".")[0] + "_stylized.jpg"
        stylized.save(stylized_path)
        print('Saving image to', stylized_path)

    # Create video from frames
    os.system(f'ffmpeg -hide_banner -loglevel error -i '
              f'{temp_folder_name}/frame_%d_stylized.jpg -q:v 2'
              f' {output_path}')

    # Remove temporary folder
    os.system(f'rm -rf {temp_folder_name}')

def stylize_image(model, image_path, output_path):
    style_model = FastStyleNet()
    style_model.load_state_dict(torch.load(model))
    style_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    style_model.to(device)

    img = Image.open(image_path)
    transform = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: x.mul(255))
    ])
    img = transform(img)
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        stylized = style_model(img).squeeze().cpu()
    stylized = stylized.clone().clamp(0, 255).numpy()
    stylized = stylized.transpose(1, 2, 0).astype("uint8")
    stylized = Image.fromarray(stylized)
    stylized.save(output_path)


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
    model = args.model
    source = args.source
    output = args.output
    if output is None:
        output = source.split('.')[0] + '_stylized.' + source.split('.')[1]

    # Video
    if source.split('.')[1] == 'mp4':
        stylize_video(model, source, output)
    # Image
    else:
        stylize_image(model, source, output)