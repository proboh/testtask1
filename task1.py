import cv2
import numpy as np
import re
from staticmap import StaticMap, CircleMarker, Line
from collections import deque

feature_params = dict(
    maxCorners=100,          # Maximum number of corners to detect
    qualityLevel=0.3,        # Minimum quality of corners
    minDistance=7,           # Minimum distance between detected corners
    blockSize=7              # Size of the averaging block for corner detection
)

# Function to extract GPS coordinates from subtitles
def extract_coordinates_from_subtitles(subtitle_file):
    gps_coordinates = []
    with open(subtitle_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # Regex to extract GPS coordinates from subtitles
            match = re.search(r'GPS \((\d+\.\d+), (\d+\.\d+)', line)
            if match:
                lon = float(match.group(1))
                lat = float(match.group(2))
                gps_coordinates.append((lat, lon))
    return gps_coordinates

# Function to generate map with the drone's path
def generate_map(lat, lon, path_coords):
    map = StaticMap(300, 300)
    path_line = Line([(lon, lat) for lat, lon in path_coords], 'blue', 3)
    map.add_line(path_line)
    marker = CircleMarker((lon, lat), 'red', 6)
    map.add_marker(marker)
    return np.array(map.render())

# Optical flow tracking function
def track_points(prev_gray, frame_gray, prev_points):
    # Calculate optical flow (i.e., track points)
    next_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_points, None)
    good_new = next_points[status == 1]
    return good_new, status

# Kalman filter for path prediction
def predict_path(points, process_noise_covariance=1e-5):
    # Kalman filter setup
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise_covariance

    # Predict path based on initial points
    predicted_path = []
    for point in points:
        measurement = np.array([[np.float32(point[0])], [np.float32(point[1])]])
        kalman.correct(measurement)
        predicted = kalman.predict()
        predicted_path.append((predicted[0], predicted[1]))
    return predicted_path

# Initialize variables
input_video = "video1.mp4"
subtitle_file = "subtitles.srt"
output_video = "drone_video_with_map.mp4"

# Extract GPS coordinates from subtitles
gps_coordinates = extract_coordinates_from_subtitles(subtitle_file)
path_coords = [(lat, lon) for lat, lon in gps_coordinates]

# Open video
cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Prepare video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# Optical Flow initialization
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
tracked_points = deque()

# Frame processing loop
frame_index = 0
gps_index = 0
last_map_image = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Track points using optical flow
    if prev_points is not None and len(prev_points) > 0:
        good_new, status = track_points(prev_gray, frame_gray, prev_points)
        tracked_points.append(good_new)

    # Update GPS data every second
    if frame_index % int(fps) == 0 and gps_index < len(gps_coordinates):
        lat, lon = gps_coordinates[gps_index]
        gps_index += 1

        # Generate map with the path
        map_image = generate_map(lat, lon, path_coords[:gps_index])

        # Resize the map image and overlay it
        map_height, map_width = 300, 300
        map_image_resized = cv2.resize(map_image, (map_width, map_height))
        last_map_image = cv2.cvtColor(map_image_resized, cv2.COLOR_RGB2BGR)

    # Overlay map if it exists
    if last_map_image is not None:
        map_height, map_width = last_map_image.shape[:2]
        frame[10:10 + map_height, -10 - map_width:-10] = last_map_image

    # Predict path using Kalman filter
    if len(tracked_points) > 0:
        predicted_path = predict_path(tracked_points[-1])

    # Write frame to output video
    out.write(frame)

    # Update previous frame and points
    prev_gray = frame_gray.copy()
    prev_points = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None

    frame_index += 1

# Release resources
cap.release()
out.release()

print(f"Video with map overlay saved as {output_video}")