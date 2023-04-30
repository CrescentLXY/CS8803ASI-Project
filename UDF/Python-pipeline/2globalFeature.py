import cv2 # use to load image, and use ORB to extract local feature
import os
import imagehash # use PHASH to extract global feature
from PIL import Image # use to convert numpy array to image object
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import struct # use to handle float number type

# Create an ORB object
orb = cv2.ORB_create()

# Path to the directory containing the images
path = '''Output/_A_basket_ball_that_has_been_sprayed_with_Vanta_black'''

# sum of the global feature (phash)
sum_pash = 0
item_count = 0
# Initialize an empty list to store phash values for each image
# phash_set = []
# # Hash function to convert a string to a numerical hash value
# def string2hash(s):
#     return hash(s) % (10 ** 8)

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
            # for descriptor in descriptors:
            #     print(descriptor.size)
            # print(filename, descriptors)
            # Obtain a 1-d global feature with PHASH
            pil_img = Image.fromarray(gray)
            phash = str(imagehash.phash(pil_img))
            # Transfer hash value (hex) to decimal (PCA only takes in numerical values)
            phash_dec = int(phash, 16)
            # print("PHASH: ", phash)
            # phash_set.append(phash_dec)
            sum_pash = sum_pash + phash_dec
            item_count = item_count + 1
        # else:
        #     print(f"Skipping non-image file: {filename}")

# print(phash_set)
# Gather global features to form a set and reduce dimension

# Convert the list of hash values to a numpy array
# group_phash_array = np.array(phash_set)

# Apply PCA to the hash values
# pca = PCA(n_components=16)
# group_feature = pca.fit_transform(group_phash_array.reshape(-1, 1))
# print(group_feature)
# list2array = np.array(list(map(string2hash, hashes_array)))
# hashes_pca = pca.fit_transform(list2array)
group_global = sum_pash/item_count
group_global_32 = struct.pack('f', group_global)
print((group_global_32, type(group_global_32)))
# print(group_global, type(group_global))

#
# print(hashes_pca.shape)
