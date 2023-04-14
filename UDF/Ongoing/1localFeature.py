import cv2 # use to load image, and use ORB to extract local feature
import os

# Create an ORB object
orb = cv2.ORB_create()

# Path to the directory containing the images
path = '''Output/_A_basket_ball_that_has_been_sprayed_with_Vanta_black'''

# Loop through all the images in the directory
for filename in os.listdir(path):
    # Check if the file is an image file
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load the image
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Extract keypoints and descriptors using ORB
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            print(filename, descriptors)
        else:
            print(f"Skipping non-image file: {filename}")
