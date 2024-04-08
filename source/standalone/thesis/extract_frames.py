import cv2
import os

def extract_frames(video_path, output_folder):
    # Open the video file
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Extract frames
    while success:
        # Save frame as image
        cv2.imwrite(os.path.join(output_folder, f"frame{count:04d}.jpg"), image)
        success, image = vidcap.read()
        count += 1

    print(f"{count} frames extracted.")

# Provide path to the video file
video_path = "/home/cxy/data/videos/experiment_2/policy_experiment2.avi"

# Provide the output folder path where you want to save the frames
output_folder = "/home/cxy/data/videos/experiment_2/output_frames"

# Call the function to extract frames
extract_frames(video_path, output_folder)