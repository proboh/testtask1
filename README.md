Task 1: Drone Path Prediction
  Objective: Using a video file (video1), extract coordinates and construct a drone’s flight path on a map.
Steps:
  1. Apply an Optical Flow algorithm (using OpenCV or any other library) to the video to track the motion of points.
  2. Based on the motion data, predict the drone’s path.
  3. Plot the path on a map for visualization.

Breakdown of Steps
1. Extract GPS Coordinates from Subtitles:
  The function extract_coordinates_from_subtitles reads the subtitle file and uses a regular expression to extract the GPS coordinates in the format GPS (latitude, longitude).
2. Track Points Using Optical Flow:
  The points from the first frame are tracked in subsequent frames using the Lucas-Kanade optical flow algorithm. The function track_points returns the new points for each frame.
3. Predict Drone's Path with Kalman Filter:
  The Kalman filter function predicts the drone's path based on the tracked points, adjusting the path for each frame.
4. Overlay Path on Map:
  A static map is created using the StaticMap library, and the path is drawn on it. The map is overlaid on each video frame using OpenCV.
5. Combine All into Video:
  The video is processed frame-by-frame, and at every second (based on FPS), the map is updated with the new GPS data. The predicted path is drawn on the video frame as well.
