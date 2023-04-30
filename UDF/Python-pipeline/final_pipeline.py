import cv2 # use to load image, and use ORB to extract local feature
import os
import faiss # use FAISS for indexing
import random
import numpy as np # for output comparison

# Create an ORB object
orb = cv2.ORB_create()

# Path to the directory containing the images
path = '''Output/_A_basket_ball_that_has_been_sprayed_with_Vanta_black'''
# path = 'tempDataset/testMix'
# path = '''Output/_Ronald_Reagan_aiming_a_hunting_rifle_on_an_Air_Force_One'''

local_feature_dimension = 32 # Set the dimensionality of the vectors to be indexed

nlist = 5 # Set the number of centroids to index for the IndexIVFFlat index

nprobe = 10 # Set the number of probes to use for the IndexIVFFlat index

# Initialize the index
quantizer = faiss.IndexFlatL2(local_feature_dimension)
index = faiss.IndexIVFFlat(quantizer, local_feature_dimension, nlist, faiss.METRIC_L2)

# Create a dictionary to map feature indices to image file names
file_dict = {}
# Another for input
test_file_dict = {}

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

            # Train the index
            index.train(descriptors)
            # Add the descriptors to the index
            index.add(descriptors)

            # Add the file name to the dictionary with the index as the key
            for i in range(descriptors.shape[0]):
                file_dict[index.ntotal - descriptors.shape[0] + i] = filename

            # Add to the test dict
            test_file_dict[filename] = gray
        else:
            print(f"Skipping non-image file: {filename}")

# Optimize the index for searching
index.nprobe = nprobe

# Choose a random image
query_file = random.choice(list(test_file_dict.keys()))
# query_file = "Output/_Ronald_Reagan_aiming_a_hunting_rifle_on_an_Air_Force_One/g1027_root.jpg"
print('Randomly select an image to test: ', query_file)

query_img = test_file_dict[query_file]

# Extract keypoints and descriptors using ORB
query_keypoints, query_descriptors = orb.detectAndCompute(query_img, None)

# Search the index for the 5 nearest neighbors of the query vector
distances, indices = index.search(query_descriptors, k=5)

# Map the feature indices to image filenames
# most_similar_files = [list(file_dict.keys())[i] for i in indices[0][1:3]]
most_similar_files = [file_dict[i] for i in indices[0][1:3]]
# most_similar_files = [file_dict[i] for i in indices[0][1:5]]

# Print the filenames of the most similar images
print("Most similar images:")
for file in most_similar_files:
    print(file) # note some condition they might be similar

# Load the query image and the most similar images
query_img = cv2.imread(os.path.join(path, query_file))
similar_imgs = [cv2.imread(os.path.join(path, file)) for file in most_similar_files]

# Create a window to display the images
cv2.namedWindow("query image and similar images", cv2.WINDOW_NORMAL)
cv2.resizeWindow("query image and similar images", 1000, 600)

# Display the query image first
cv2.imshow("query image and similar images", query_img)
cv2.waitKey(0)
# Display each similar image one by one
for img in similar_imgs:
    if img is not None and not np.array_equal(img, query_img):
        cv2.imshow("query image and similar images", img)
        cv2.waitKey(0)
cv2.destroyAllWindows()

