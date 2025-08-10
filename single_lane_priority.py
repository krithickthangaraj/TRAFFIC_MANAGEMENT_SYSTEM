import cv2
import multiprocessing
import time
import numpy as np

# Load YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Global sorted keys
previous_cycle = []
sorted_keys = []

def countdown_timer(seconds, lane, pixel_area, previous_cycle, shared_new_order):
    while seconds > 0:
        if seconds < seconds // 2:
            previous_cycle.append(lane)
            shared_new_order.pop(0)
        time.sleep(1)
        seconds -= 1
    print(f"Lane {lane} timer completed.")

def calculate_timer_duration(pixel_area, min_time=10, max_time=30):
    min_area = 10000
    max_area = 500000
    normalized_area = (pixel_area - min_area) / (max_area - min_area)
    return int(min_time + (max_time - min_time) * normalized_area)

def detect_vehicles_and_calculate_area(frames):
    results = {}
    vehicle_classes = ["car", "truck", "bus", "motorbike"]

    for idx, frame in enumerate(frames):
        lane_key = f"lane{idx + 1}"
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        total_area = 0
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    area = w * h
                    total_area += area

        results[lane_key] = total_area
    return dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

def new_cycle(sorted_keys, previous_cycle):
    new_order = []
    while previous_cycle or sorted_keys:
        if previous_cycle:
            popped_element = previous_cycle.pop(0)
            new_order.append(popped_element)
        if sorted_keys:
            popped_element = sorted_keys.pop(0)
            new_order.append(popped_element)
    return new_order

def extract_frame_at_time(video_capture, time_in_seconds):
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_number = int(fps * time_in_seconds)
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = video_capture.read()
    if not ret:
        raise Exception(f"Could not read frame at {time_in_seconds}s")
    return frame

def vehicle_detection(shared_new_order, shared_previous_cycle):
    video_paths = [
        "static/videos/bus.mp4",
        "static/videos/car.mp4",
        "static/videos/bus.mp4",
        "static/videos/car.mp4"
    ]
    video_captures = [cv2.VideoCapture(video_path) for video_path in video_paths]

    while True:
        frames = [extract_frame_at_time(vc, 0) for vc in video_captures]  # Extract the first frame from each video
        sorted_total_areas = detect_vehicles_and_calculate_area(frames)
        sorted_keys = list(sorted_total_areas.keys())
        shared_new_order[:] = new_cycle(sorted_keys, shared_previous_cycle)
        shared_previous_cycle[:] = []

        while shared_new_order:
            lane = shared_new_order[0]
            pixel_area = sorted_total_areas[lane]
            timer_duration = calculate_timer_duration(pixel_area)
            countdown_timer(timer_duration, lane, pixel_area, shared_previous_cycle, shared_new_order)

if __name__ == '__main__':
    with multiprocessing.Manager() as manager:
        shared_new_order = manager.list()
        shared_previous_cycle = manager.list()

        process1 = multiprocessing.Process(target=vehicle_detection, args=(shared_new_order, shared_previous_cycle))
        process1.start()
        process1.join()
