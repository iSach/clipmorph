from flask import Flask, url_for, redirect, render_template, request, send_file
from moviepy.editor import VideoFileClip
import os
from werkzeug.exceptions import BadRequestKeyError

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.static_folder = 'static'

def inverse_color_video(input_file, output_file, style):
    """Invert the colors of a video."""
    print('Inverting colors of video; style: ', style)
    clip = VideoFileClip(input_file)
    inverted_clip = clip.fl_image(lambda image: 255 - image)
    inverted_clip.write_videofile(output_file)
    clip.close()
    inverted_clip.close()

def get_style_options():
    with open('styles.txt', 'r') as file:
        return [line.strip() for line in file]

@app.route('/')
def index():
    return render_template('index.html', style_options=get_style_options())

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        try:
            style = request.form['style-select']
            print('Style selected: ', style)
        except BadRequestKeyError:
            return 'No style selected', 400

        file = request.files['file']
        if file and style:
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Generate path for modified video
            modified_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'modified_' + file.filename)

            # Invert colors and save modified video
            inverse_color_video(file_path, modified_file_path, style)

            # Provide a downloadable link for the modified video file
            return send_file(
                modified_file_path,
                as_attachment=True
            )
if __name__ == '__main__':
    app.run(debug=True)
