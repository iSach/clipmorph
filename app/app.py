from flask import (Flask, url_for, redirect, render_template, request,
                   send_file, after_this_request)
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

def get_style_options():
    style_dir = '../models/'
    return [f.split("/")[-1].split(".")[0] for f in os.listdir(style_dir) if
            f.endswith('.pth')]

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

        style = os.path.join('../models', style + '.pth')

        file = request.files['file']
        if file and style:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            output_path = file_path.split('.')[0] + '_output.mp4'
            stylize_video(style, file_path, output_path, batch_size=16)

            @after_this_request
            def remove_file(response):
                try:
                    os.remove(file_path)
                    os.remove(output_path)
                except Exception as error:
                    app.logger.error(
                        "Error removing or closing downloaded file handle",
                        error)
                return response

            return send_file(
                output_path,
                as_attachment=True
            )

if __name__ == '__main__':
    app.run(debug=True)