from collections import OrderedDict
import numpy as np
import cv2
from flask import Flask, render_template, Response, jsonify
import time
from scipy.spatial import distance as dist

# Centroid tracker class to track unique objects
class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, detections):
        if len(detections) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(detections), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(detections):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            for col in unusedCols:
                self.register(inputCentroids[col])

        return self.objects

tracker = CentroidTracker()

app = Flask(__name__)

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

traffic_data = {
    "cars_in_traffic": 0,
    "level_of_service": 0,
    "total_today": 0,
    "category_distribution": {
        "car": 0,
        "truck": 0,
        "bus": 0,
        "motorbike": 0
    },
    "delay_time_stats": []
}

video_path = 'static/videos/bus.mp4'
vehicle_count = 0  # Total count of vehicles
unique_vehicle_ids = set()  # Track unique vehicle IDs


#single_lane_priority

video_paths = {
    'lane1': 'static/videos/bus.mp4',
    'lane2': 'static/videos/car.mp4',
    'lane3': 'static/videos/bus.mp4',
    'lane4': 'static/videos/car.mp4'
}

def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        success, frame = cap.read()
        if not success:
            break
        # Process the frame (detection logic can be added here)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate_frames():
    global traffic_data, vehicle_count, unique_vehicle_ids
    cap = cv2.VideoCapture(video_path)
    frame_counter = 0
    delay_times = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read frame")
            break

        frame_counter += 1
        height, width, channels = frame.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        car_count = 0
        category_counts = {"car": 0, "truck": 0, "bus": 0, "motorbike": 0}

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

                    label = str(classes[class_id])
                    if label in category_counts:
                        category_counts[label] += 1
                    if label == "car":
                        car_count += 1

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        detections = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                # Store detection (x, y, x+w, y+h)
                detections.append((x, y, x + w, y + h))

        # Update centroid tracker and vehicle count
        objects = tracker.update(detections)

        # Count vehicles based on unique object IDs
        for objectID in objects.keys():
            if objectID not in unique_vehicle_ids:
                vehicle_count += 1  # Only increment for new unique IDs
                unique_vehicle_ids.add(objectID)

        # Update real-time data
        traffic_data["cars_in_traffic"] = len(objects)  # Active vehicles in frame
        traffic_data["total_today"] = vehicle_count  # Total unique vehicles today
        traffic_data["category_distribution"] = category_counts
        delay_times.append(car_count)

        if len(delay_times) > 50:
            delay_times = delay_times[-50:]

        traffic_data["delay_time_stats"] = delay_times

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/single_lane_priority')
def manual_control():
    return render_template('single_lane_priority.html')



@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def get_data():
    return jsonify(traffic_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)
