import argparse
import os

import torch
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from clipmorph.data import load_data
from clipmorph.nn import FastStyleNet

import ffmpeg


def stylize_video(model, video_path, output_path, batch_size=16, socketio=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_name = video_path.split("/")[-1].split(".")[0]

    # Create temporary folder for frames
    temp_folder_name = f"temp_{video_name}"
    os.makedirs(temp_folder_name, exist_ok=True)

    # Extract frames from video
    os.system(
        f'ffmpeg -y -hide_banner -loglevel error -i "{video_path}" -q:v 2'
        f" {temp_folder_name}/frame_%d.jpg"
    )

    style_model = FastStyleNet()
    style_model.load_state_dict(
        torch.load(model, map_location=device),
        strict=False,
    )
    style_model.eval()
    if device == torch.device("cuda"):
        style_model.to(device)

    data = load_data(temp_folder_name, batch_size)
    num_images = data.dataset.num_images

    progress_bar = tqdm(total=num_images, desc="Stylizing images")

    for img, names in data:
        if device == torch.device("cuda"):
            img = img.to(device)

        # Stylize
        with torch.no_grad():
            stylized = style_model(img).cpu()

        # Save
        for i, stylized_img in enumerate(stylized):
            stylized_img = stylized_img.clone().clamp(0, 255).numpy()
            stylized_img = stylized_img.transpose(1, 2, 0).astype("uint8")
            stylized_img = Image.fromarray(stylized_img)
            stylized_img_path = (
                temp_folder_name
                + "/"
                + names[i].split("/")[-1].split(".")[0]
                + "_stylized.jpg"
            )
            stylized_img.save(stylized_img_path)

        del img
        del stylized

        progress_bar.update(len(names))
        if socketio:
            socketio.sleep(0)
            socketio.emit("progress", {"current": progress_bar.n, "total": num_images})

    # Create video from frames
    probe = ffmpeg.probe(video_path)
    video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
    fps = int(video_info["r_frame_rate"].split("/")[0])
    os.system(
        f'ffmpeg -y -hide_banner -loglevel error -i "{video_path}" -i '
        f'{temp_folder_name}/frame_%d_stylized.jpg -framerate {fps} -map "0:a?" -map 1:v -c:v libx264 -c:a copy -q:v 2'
        f' "{output_path}"'
    )

    # Remove temporary folder
    os.system(f"rm -rf {temp_folder_name}")


def stylize_image(model, image_path, output_path, socketio=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    style_model = FastStyleNet()
    style_model.load_state_dict(
        torch.load(model, map_location=device),
        strict=False,
    )
    style_model.eval()
    style_model.to(device)

    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor(), T.Lambda(lambda x: x.mul(255))])
    img = transform(img)
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        stylized = style_model(img).squeeze().cpu()
    stylized = stylized.clone().clamp(0, 255).numpy()
    stylized = stylized.transpose(1, 2, 0).astype("uint8")
    stylized = Image.fromarray(stylized)
    if socketio:
        socketio.emit("progress", {"current": 1, "total": 1})
    stylized.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a pre-trained model.")
    parser.add_argument(
        "--model",
        type=str,
        default="models/gericault.pth",
        help="Path to the model .pth file",
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Path of the image/video to stylize",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path of the output image/video",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for video stylization",
    )

    args = parser.parse_args()
    model = args.model
    source = args.source
    output = args.output
    if output is None:
        model_name = model.split("/")[-1].split(".")[0]
        output = source.split(".")[0] + f"_{model_name}." + source.split(".")[1]

    # Video
    if source.split(".")[1] == "mp4":
        stylize_video(model, source, output, batch_size=args.batch_size)
    # Image
    else:
        stylize_image(model, source, output)
