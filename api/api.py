from flask import Flask, url_for, redirect, render_template, request, send_file
import os
from werkzeug.exceptions import BadRequestKeyError
from clipmorph.run import stylize_video
import threading 

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.static_folder = 'static'

def inverse_color_video(input_file, style):
    """Invert the colors of a video."""
    print('Applying the style: ', style)
    output_path = input_file.split('.')[0] + '_output.mp4'
    stylize_video(style, input_file, output_path, batch_size=16)
    return output_path

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

            # Invert colors and save modified video
            modified_file_path = inverse_color_video(file_path, style)

            # Provide a downloadable link for the modified video file
            return send_file(
                modified_file_path,
                as_attachment=True
            )
if __name__ == '__main__':
    app.run(debug=True)
