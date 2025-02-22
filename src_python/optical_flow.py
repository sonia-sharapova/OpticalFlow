'''
Sequential Optical Flow Estimation from DICOM Files

More dtail in README.md and ScientificComputingFinal.pdf
References:
(also in README.md)
Load DICOM files: https://pydicom.github.io/pydicom/stable/tutorials/dataset_basics.html
'''
import os
import cv2
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

# Fine-tuned parameters for Shi-Tomasi
SHI_TOMASI_PARAMS = {
    "maxCorners": 150,      # Maximum number of corners
    "qualityLevel": 0.03,   # Minimum accepted quality of corners
    "minDistance": 7        # Minimum distance between detected corners
}

# Fine-tuned parameters for Lucas-Kanade
LUCAS_KANADE_PARAMS = {
    "winSize": (15, 15),    # Size of the window for optical flow
    "maxLevel": 3,          # Number of pyramid levels
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)  # Termination criteria
}

# Open and read the DICOM files
def load_dicom_files(directory):
    '''
    Loads valid DICOM files from a specified directory.
    '''
    dicom_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".dcm") and not file.startswith("._"):  # Skip hidden macOS files
                dicom_files.append(os.path.join(root, file))
    return sorted(dicom_files)  # Sort files for sequential processing

def preprocess_dicom(dicom_file_path, fixed_size=(256, 256)):
    '''
    Preprocesses DICOM files 
        - ensure consistent dimensions and contrast.
    '''
    try:
        ds = pydicom.dcmread(dicom_file_path, force=True)
        if not hasattr(ds, "pixel_array"):
            print(f"No pixel array found in {dicom_file_path}. Skipping this file.")
            return None
        pixel_array = ds.pixel_array
        pixel_array = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        img_eq = cv2.equalizeHist(pixel_array)
        img_resized = cv2.resize(img_eq, fixed_size)
        return img_resized
    except Exception as e:
        print(f"Error reading {dicom_file_path}: {e}")
        return None

# Feature detection function
def detect_features(image):
    '''Detects Shi-Tomasi corners in the image'''
    corners = cv2.goodFeaturesToTrack(image, **SHI_TOMASI_PARAMS)
    return np.float32([f.ravel() for f in corners]) if corners is not None else []

# Optical flow computation
def compute_optical_flow(prev_img, next_img, features):
    '''Computes Lucas-Kanade optical flow'''
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_img, next_img, features, None, **LUCAS_KANADE_PARAMS)
    return next_pts, status


def overlay_optical_flow(image, features, flow_vectors):
    '''Creates a frame with optical flow vectors overlaid'''
    image_copy = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
    for (x1, y1), (x2, y2) in zip(features, flow_vectors):
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        cv2.arrowedLine(image_copy, pt1, pt2, color=(0, 255, 0), thickness=1)
    return cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

def save_results_as_gif(frames, output_path, duration=500):
    '''Saves a list of frames as an animated GIF'''
    pil_images = [Image.fromarray(frame) for frame in frames]
    pil_images[0].save(
        output_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration,
        loop=0
    )
    print(f"GIF saved at {output_path}")

def process_dataset(dataset_dir, output_dir):
    '''
    Processes all datasets in the directory and saves separate GIFs for each subfolder
    '''
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder}")
            dicom_files = load_dicom_files(folder_path)
            if len(dicom_files) < 2:
                print(f"Not enough frames in {folder} to compute optical flow.")
                continue
            
            # Preprocess frames
            images = [preprocess_dicom(dcm) for dcm in dicom_files]
            images = [img for img in images if img is not None]
            if len(images) < 2:
                print(f"Not enough valid frames in {folder} after preprocessing.")
                continue
            
            # Compute optical flow and generate frames
            frames = []
            for i in range(len(images) - 1):
                prev_img = images[i]
                next_img = images[i + 1]
                features = detect_features(prev_img)
                if len(features) == 0:
                    print(f"No features detected in frame {i}. Skipping optical flow computation.")
                    continue
                
                next_pts, status = compute_optical_flow(prev_img, next_img, features)
                good_prev_pts = features[status.flatten() == 1]
                good_next_pts = next_pts[status.flatten() == 1]
                
                # Overlay optical flow vectors
                frame = overlay_optical_flow(next_img, good_prev_pts, good_next_pts)
                frames.append(frame)

            # Create subfolder-specific output directory
            folder_output_dir = os.path.join(output_dir, folder)
            if not os.path.exists(folder_output_dir):
                os.makedirs(folder_output_dir)

            # Save GIF in the folder's output directory
            gif_path = os.path.join(folder_output_dir, f"{folder}.gif")
            if frames:
                save_results_as_gif(frames, gif_path)



if __name__ == '__main__':
    # Path to directory containing DICOM files (can be in subdirectories)
    parent_dir = sys.argv[1]
    folder_name = parent_dir.split("/")[-1]
    output_directory = f"./output/{folder_name}"

    process_dataset(parent_dir, output_directory)