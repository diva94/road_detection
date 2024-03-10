import pandas as pd
from joblib import load
import io
from flask import Flask, render_template, request, session, send_file
from flask_session import Session
from tempfile import mkdtemp
import os
from inference import run
from fpdf import FPDF

app = Flask(__name__)

# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# allowed input image extensions
IMG_EXTENSIONS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo',
'JPG']

# allowed input video extensions
VID_EXTENSIONS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']


def detect(test_input):
    """ 
    Function to detect potholes given an image input
    Parameters:
        test_input: image/video input
    Output:
        image/video with detection
    """
    # weights
    weights_path = 'best.pt'
    # to save the detection
    # output_path = 'static'
    # img dimension
    count=0
    # img_size = [img_width, img_height]
    # # confidence threshold
    # conf_threshold = 0.5
    # # iou threshold
    # iou_threshold = 0.5
    # # bounding box thickness
    # bbox_line_thick = 5

    # run detection
    run(test_input,weights_path,count)

# home page
@app.route("/")
def index():
    return render_template(
        "index.html", 
        ori_image="static/orig.jpg", 
        det_image="static/pred.jpg",
        fName=None
        )


# carryout detection
@app.route("/detection", methods=["GET", "POST"])
def detection():
    file = request.files["test_file"]

    # check if a file is uploaded
    if file:
        # check file extention
        ext = file.filename.split('.')[-1]

        # detection on an image
        if ext in IMG_EXTENSIONS:
            # create a path
            file_path = os.path.join("static/test", file.filename)
            # save the file
            file.save(file_path)
            # carryout detection
            detect(file_path)

            return render_template(
                "index.html", type="primary", 
                message="Done!, Image is ready to download", 
                ori_image=f"{file.filename}", 
                det_image=f"{file.filename}",
                fName=file.filename
            )
        # detection on a video
        elif ext in VID_EXTENSIONS:
            # create a path
            file_path = os.path.join("static", file.filename)
            # save the file
            file.save(file_path)
            detect(file_path)
            return render_template(
                "index.html", type="primary", 
                message="Done!, Video is ready to download.",
                ori_image="static/orig.jpg", 
                det_image="static/pred.jpg",
                fName=file.filename
            )
    # if no input render an error message        
    else:
        return render_template(
                "index.html", type="danger", 
                message="Please upload a file.",
                ori_image="static/orig.jpg", 
                det_image="static/pred.jpg",
                fName=None
            )

# download the detection
@app.route("/download/<fName>", methods=["GET", "POST"])
def download(fName):
    det_path = f"output/{fName}"
    return send_file(det_path, as_attachment=True)

# download error handling
@app.route("/download_error", methods=["GET", "POST"])
def download_error():
    return render_template(
            "index.html", type="danger", 
            message="Please upload a file and carryout detection.",
            ori_image="static/orig.jpg", 
            det_image="static/pred.jpg",
            fName=None
        )

@app.route("/generate_pdf")
def generate_pdf():
    object_counts = {
    'potholes': 10,
    'road_furniture': 25,
    'traffic_lights': 5,
    }
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Object Detection Report", ln=True, align="C")
    for obj, count in object_counts.items():
        pdf.cell(200, 10, txt=f"{obj}: {count}", ln=True, align="L")
    pdf.output("report.pdf")
    return send_file("report.pdf", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host="0.0.0.0", port=5000)