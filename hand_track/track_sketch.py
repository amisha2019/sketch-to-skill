
import cv2
import imageio
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter, ImageDraw
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Check if frames were successfully loaded
    if len(frames) == 0:
        print(f"Error: No frames found in the video file {video_path}")
    
    return frames

def hand_segmentation(frame):
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Create a blank mask to draw hands with 3 channels
    mask = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    hand_landmarks_positions = []  # To store hand landmark positions for tracking

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Using default styles by removing the style arguments
            mp.solutions.drawing_utils.draw_landmarks(
                image=mask,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS)
            
            # Extract hand's landmark coordinates for tracking
            for landmark in hand_landmarks.landmark:
                h, w, _ = frame.shape
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                hand_landmarks_positions.append((x, y))

    # Convert mask back to single channel for further processing if needed
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask, hand_landmarks_positions

def generate_sketch(image):
    pil_image = Image.fromarray(image).convert("L")
    pil_image = pil_image.filter(ImageFilter.FIND_EDGES)
    return np.array(pil_image)

def save_image(image, folder, filename):
    """ Saves an image to disk. """
    path = os.path.join(folder, filename)
    cv2.imwrite(path, image)

def process_video(frames, output_folder):
    if len(frames) == 0:
        print("No frames to process.")
        return []

    sketches = []
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Create a blank image to draw the trajectory
    trajectory_image = np.zeros_like(frames[0])

    previous_point = None  # Keep track of the previous hand position
    
    for idx, frame in enumerate(frames):
        mask, hand_positions = hand_segmentation(frame)
        
        # Ensure mask is in correct format (CV_8U and same size as frame)
        if len(mask.shape) != 2 or mask.dtype != np.uint8:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Use mask to segment the frame
        segmented_image = cv2.bitwise_and(frame, frame, mask=mask)
        sketch = generate_sketch(segmented_image)
        sketches.append(sketch)

        # Track and draw the hand's trajectory using landmarks
        if hand_positions:
            # Use the first hand's wrist position (landmark 0) as the tracking point
            wrist_position = hand_positions[0]
            if previous_point:
                # Draw a line from the previous position to the current wrist position
                cv2.line(trajectory_image, previous_point, wrist_position, (0, 255, 0), 2)  # Green line
            previous_point = wrist_position
        
        # Overlay the trajectory on the current frame
        combined_image = cv2.addWeighted(frame, 0.7, trajectory_image, 0.3, 0)

        ### SHOW INTERMEDIATE RESULTS ###
        # # Show the segmentation mask
        # cv2_imshow(mask)
        
        # # Show the trajectory as it progresses
        # cv2_imshow(trajectory_image)

        # # Show the current frame with trajectory overlay
        # cv2_imshow(combined_image)

        # # Pause between frames to observe output
        # if cv2.waitKey(50) & 0xFF == ord('q'):
        #     break

        # Convert frame from RGB to BGR for saving as OpenCV uses BGR format
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
        
        # Save the segmented image and the sketch
        save_image(frame_bgr, output_folder, f'original_{idx}.png')
        save_image(segmented_image_bgr, output_folder, f'segmented_{idx}.png')
        save_image(sketch, output_folder, f'sketch_{idx}.png')
        
        # Save the image with the trajectory overlay
        save_image(combined_image, output_folder, f'trajectory_{idx}.png')

    return sketches

# New GIF creation function
def create_gif(image_folder, gif_name):
    """ Create a GIF from a sequence of images in the given folder """
    images = sorted([img for img in os.listdir(image_folder) if img.startswith('trajectory_') and img.endswith(".png")])
    
    # Check if there are any images to create the GIF
    if len(images) == 0:
        print(f"No images found starting with 'trajectory_' in {image_folder}")
        return
    
    # Load all images into a list
    frames = [Image.open(os.path.join(image_folder, image)) for image in images]
    
    # Save the list of images as a GIF
    frames[0].save(gif_name, format='GIF', append_images=frames[1:], 
                   save_all=True, duration=100, loop=0)
    print(f"GIF saved at {gif_name}")

def main():
    video_path = 'hand_track/video.mp4'  # Update with your correct path
    output_folder = 'hand_track/output_images'
    
    # Load video frames
    frames = load_video(video_path)

    # Process video if frames were loaded successfully
    if frames:
        sketches = process_video(frames, output_folder)
    
        # Create GIF after processing all the frames
        gif_name = 'hand_track/hand_trajectory.gif'
        create_gif(output_folder, gif_name)
    
        # Display the first sketch for demonstration
        if sketches:
            plt.imshow(sketches[0], cmap='gray')
            plt.show()

if __name__ == "__main__":
    main()
