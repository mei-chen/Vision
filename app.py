from flask import Flask, request, render_template, json
from flask_cors import CORS, cross_origin
import tempfile
import os
import detector

app = Flask(__name__)


@app.route("/")
def home():
    return "ok"


@app.route("/demo")
def demo():
    return render_template('demo.html')


@app.route("/api/v1/detect", methods=['POST'])
@cross_origin()
def detect():
    upload = request.files.getlist("file")[0]
    print("File name: {}".format(upload.filename))
    filename = upload.filename

    _, fn = tempfile.mkstemp()
    upload.save(fn)
    res = detector.run(fn)
    os.remove(fn)

    return json.jsonify(res)


if __name__ == "__main__":
    app.run(host='0.0.0.0')
