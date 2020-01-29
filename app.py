from flask import Flask, render_template, send_from_directory, Response
from pathlib import Path
from capture import capture_and_save
from camera import Camera
import argparse

camera = Camera()
camera.run()


app = Flask(__name__)
# app.config["SECRET_KEY"] = "secret!"


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering or Chrome Frame,
    and also to cache the rendered page for 10 minutes
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers["Cache-Control"] = "public, max-age=0"
    return r


@app.route("/")
def entrypoint():
    return render_template("index.html")


@app.route("/capture")
def capture():
    im = camera.get_frame(bytes=False)
    capture_and_save(im)
    return render_template("send_to_init.html")


@app.route("/images/last")
def last_image():
    p = Path("images/last.png")
    if p.exists():
        r = "last.png"
        gender = camera.gender
        age = camera.age
        emotion = camera.emotion
    else:
        print("No last")
        r = "not_found.jpeg"
    return send_from_directory("images",r)


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')

def get_result(camera):
    emotion = camera.emotion
    gender = camera.gender
    age = camera.age
    return emotion, gender, age


@app.route("/stream")
def stream_page():
    gender = "..."
    age = "..."
    emotion = "..."
    return render_template("stream.html", gender = gender, age = age, emotion = emotion)


@app.route("/get_prediction")
def get_prediction():
    emotion, gender, age = get_result(camera)
    return render_template("stream.html", emotion=emotion, gender=gender, age=age)


@app.route("/video_feed")
def video_feed():
    return Response(gen(camera),
        mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--port',type=int,default=5000, help="Running port")
    parser.add_argument("-H","--host",type=str,default='0.0.0.0', help="Address to broadcast")
    args = parser.parse_args()
    app.run(host=args.host,port=args.port)
