# Import necessary modules for vehicle detection using OpenCV
import cv2
import numpy as np
from flask import Flask, Response

app = Flask(__name__)

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Flask route to serve video frame
@app.route('/frame')
def frame():
    # Your vehicle detection code
    # ...
    # Return processed frame
    ret, jpeg = cv2.imencode('.jpg', frame)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(port=5001)
