import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def hand_segmentation(frame):
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Create a blank mask to draw hands with 3 channels
    mask = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Using default styles by removing the style arguments
            mp.solutions.drawing_utils.draw_landmarks(
                image=mask,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS)
    # Convert mask back to single channel for further processing if needed
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask



def generate_sketch(image):
    pil_image = Image.fromarray(image).convert("L")
    pil_image = pil_image.filter(ImageFilter.FIND_EDGES)
    return np.array(pil_image)

def save_image(image, folder, filename):
    """ Saves an image to disk. """
    path = os.path.join(folder, filename)
    cv2.imwrite(path, image)

def process_video(frames, output_folder):
    sketches = []
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for idx, frame in enumerate(frames):
        mask = hand_segmentation(frame)

        # Ensure mask is in correct format (CV_8U and same size as frame)
        if len(mask.shape) != 2 or mask.dtype != np.uint8:
            # Convert mask to grayscale if it's not
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.shape != frame.shape[:2]:
            # Resize mask to match frame size if necessary
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Use mask to segment the frame
        segmented_image = cv2.bitwise_and(frame, frame, mask=mask)
        sketch = generate_sketch(segmented_image)
        sketches.append(sketch)
        
        # Convert frame from RGB to BGR for saving as OpenCV uses BGR format
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
        
        # Save the segmented image and the sketch
        save_image(frame_bgr, output_folder, f'original_{idx}.png')
        save_image(segmented_image_bgr, output_folder, f'segmented_{idx}.png')
        save_image(sketch, output_folder, f'sketch_{idx}.png')
    
    return sketches


def main():
    video_path = 'hand_track/20240918_102018.mp4'
    output_folder = 'hand_track/output_images'
    frames = load_video(video_path)
    sketches = process_video(frames, output_folder)
    
    # Display the first sketch for demonstration
    plt.imshow(sketches[0], cmap='gray')
    plt.show()

if __name__ == "__main__":
    main()
