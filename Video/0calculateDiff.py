import cv2
import numpy as np
# from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as compare_ssim

# Read the video file
cap = cv2.VideoCapture('testVideo.mp4')

# Read the first frame
ret, frame1 = cap.read()

# Convert the frame to grayscale
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Loop through the video frames
while True:
    # Read the next frame
    ret, frame2 = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute the mean square error (MSE) between the two frames
    mse = np.mean((gray1 - gray2) ** 2)

    # Compute the structural similarity index (SSIM) between the two frames
    ssim = compare_ssim(gray1, gray2)

    # Print the similarity metrics
    print(f"MSE: {mse}, SSIM: {ssim}")

    # Update the previous frame
    gray1 = gray2

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
