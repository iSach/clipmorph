
def video_to_frames(input_loc, output_loc):
    """
    Extract frames from input video file
    and save them as separate frames in an output directory.

    Reference: https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames

    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.

    Returns:
        fps : frame rate of the source video
    """
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            continue
        # Write the results back to output location.
        cv2.imwrite(output_loc + str(count) + ".jpg", frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_lengt h -1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print \
                ("It took %d seconds forconversion." % (time_end - time_start))
            break
    return fps


def frames_to_video(input_loc, output_loc, fps):
    img_array = []
    img_list = os.listdir(input_loc)
    for i in range(len(img_list)):
        path = input_loc + str(i) + ".jpg"
        img = cv2.imread(path)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(output_loc, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                          size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def style(path_to_original, style_model):
    import os
    model_dir = "./models/" + style_model + ".pth"
    if not os.path.exists(
            "frames_video"):  # Path to the model you use to stylize your file
        os.makedirs("frames_video")
    content_dir = "frames_video"  # Path to which the original video frames will be extract. (If image, does not matter)
    if not os.path.exists("output"):
        os.makedirs("output")
    output_dir = "output"  # Path to which the stylized video frames will be put
    if not os.path.exists("result"):
        os.makedirs("result")
    stylized_dir = "result"  # Path to the final stylized file

    print('--------------------- Stylizing the file: ', output_dir)

    # First delete all files in the content_dir
    import os
    for f in os.listdir(content_dir):
        os.remove(content_dir + f)

    # Extract frames if gif or video
    name, ext = os.path.splitext(os.path.basename(path_to_original))

    if ext == '.jpg':
        style_model = FastStyleNet()
        style_model.load_state_dict(torch.load(model_dir))
        style_model.to(device)
        img_path = path_to_original
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
        from PIL import Image
        stylized = Image.fromarray(stylized)
        stylized.save(stylized_dir + path_to_original.split('/')[-1])
        print('Image stylized.')
    else:
        if ext == '.gif':
            from PIL import Image
            imageObject = Image.open(path_to_original)
            for frame in range(0, imageObject.n_frames):
                imageObject.seek(frame)
                imageObject.convert('RGB').save(
                    content_dir + str(frame) + ".jpg")
        elif ext == '.mp4':
            fps = video_to_frames(path_to_original, content_dir)

        # Delete all files in the output_dir and apply stylization
        for f in os.listdir(output_dir):
            os.remove(output_dir + f)
        neural_style_transfer(model_dir, content_dir, output_dir)

        # Reconstruct gif or video and put the new stylized file in 'stylized_dir'
        if ext == '.gif':
            frames = os.listdir(output_dir)
            GIF_list = []
            for i in range(len(frames)):
                new_frame = Image.open(output_dir + str(i) + ".jpg")
                GIF_list.append(new_frame)
            # Save into a GIF
            GIF_list[0].save(stylized_dir + name + '.gif', format='GIF',
                             append_images=GIF_list[1:],
                             save_all=True, )
        elif ext == '.mp4':
            frames_to_video(output_dir, stylized_dir + name + '.mp4', fps)
    # delete folder and its content
    import shutil
    shutil.rmtree(content_dir)
    shutil.rmtree(output_dir)

    return os.path.join(stylized_dir, name + '.mp4')
