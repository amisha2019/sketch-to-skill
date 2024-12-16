import cv2
import mediapipe as mp
import numpy as np
import imageio

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

def draw_sketch_trajectory(frame_shape, points):
    """Draw trajectory on a blank canvas to look like a sketch."""
    sketch = np.full((frame_shape[0], frame_shape[1], 3), 255, np.uint8)  # White background
    if points:
        for i in range(1, len(points)):
            cv2.line(sketch, points[i - 1], points[i], (0, 0, 0), 3)  # Black line for trajectory
    return sketch

def track_hand_center_and_draw_sketch(frames):
    center_points = []
    sketches = []
    for idx, frame in enumerate(frames):
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if result.multi_hand_landmarks:
            x_coords = [int(lm.x * frame.shape[1]) for lm in result.multi_hand_landmarks[0].landmark]
            y_coords = [int(lm.y * frame.shape[0]) for lm in result.multi_hand_landmarks[0].landmark]
            bbox = (min(x_coords), max(x_coords), min(y_coords), max(y_coords))
            cx = (bbox[0] + bbox[1]) // 2
            cy = (bbox[2] + bbox[3]) // 2
            center_points.append((cx, cy))
            print(f"Frame {idx}: Hand detected. Center: ({cx}, {cy})")
        else:
            print(f"Frame {idx}: No hand landmarks detected.")
        sketch = draw_sketch_trajectory(frame.shape, center_points)
        sketches.append(sketch)
    return sketches

def create_gif(images, output_path, fps=10):
    print(f"Creating GIF with {len(images)} frames.")
    with imageio.get_writer(output_path, mode='I', fps=fps) as writer:
        for image in images:
            writer.append_data(image)

def main():
    video_path = 'hand_track/video.mp4'
    frames = load_video(video_path)
    print(f"Loaded {len(frames)} frames from the video.")
    sketches = track_hand_center_and_draw_sketch(frames)
    final_sketch = sketches[-1] if sketches else np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.imwrite('final_sketch_trajectory.png', final_sketch)
    if sketches:
        create_gif(sketches, 'sketch_trajectory.gif')
    else:
        print("No sketches available to create a GIF.")

if __name__ == "__main__":
    main()
