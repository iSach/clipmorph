from flask import Flask, render_template, request, send_file, after_this_request
import os
from werkzeug.exceptions import BadRequestKeyError
from clipmorph.run import stylize_video, stylize_image
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.static_folder = "static"


def get_style_options():
    # TODO: for a more automated approach, we could use buckets in GCS
    #       and with a nice pipeline have the trained automatically uploaded
    #       there
    style_dir = "../models/"
    styles = []
    for f in os.listdir(style_dir):
        if f.endswith(".pth"):
            style_name = f.split(".")[0]
            image_path = f"static/styles/{style_name}.jpg"
            styles.append((style_name, image_path))
    return styles


@app.route("/")
def index():
    return render_template("index.html", style_options=get_style_options())


@app.route("/upload", methods=["POST"])
def upload_file():
    if request.method == "POST":
        try:
            style = request.form["style-select"]
            print("Style selected: ", style)
        except BadRequestKeyError:
            return "No style selected", 400

        style = os.path.join("../models", style + ".pth")

        file = request.files["file"]
        if file and style:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            extension = file_path.split(".")[-1]

            output_path = file_path.split(".")[0] + "_output." + extension

            if extension == "mp4":
                stylize_video(
                    style, file_path, output_path, batch_size=16, socketio=socketio
                )
            else:
                stylize_image(style, file_path, output_path, socketio=socketio)

            @after_this_request
            def remove_file(response):
                try:
                    os.remove(file_path)
                    os.remove(output_path)
                except Exception as error:
                    app.logger.error(
                        "Error removing or closing downloaded file handle", error
                    )
                return response

            return send_file(output_path, as_attachment=True)


if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=8080, allow_unsafe_werkzeug=True)
