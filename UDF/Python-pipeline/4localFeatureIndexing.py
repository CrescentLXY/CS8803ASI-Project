import cv2 # use to load image, and use ORB to extract local feature
import os
import imagehash # use PHASH to extract global feature
from PIL import Image # use to convert numpy array to image object
import numpy as np
from sklearn.decomposition import PCA
import struct # use to handle float number type
import faiss # use FAISS for indexing

# Create an ORB object
orb = cv2.ORB_create()

# Path to the directory containing the images
path = '''Output/_A_basket_ball_that_has_been_sprayed_with_Vanta_black'''

# Set the dimensionality of the vectors to be indexed
local_feature_dimension = 32

# Set the number of centroids to index for the IVFADC8 index
nlist = 100

# Set the number of probes to use for the IVFADC8 index
nprobe = 10

# Initialize the index
quantizer = faiss.IndexFlatL2(local_feature_dimension)
index = faiss.IndexIVFFlat(quantizer, local_feature_dimension, nlist, faiss.METRIC_L2)

nsubquantizer = 8
nbsubquantizerindex = 8

# index = faiss.IndexIVFPQ(quantizer, local_feature_dimension, nlist, nsubquantizer, nbsubquantizerindex)

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
            print('descriptors', type(descriptors), len(descriptors))

            # # Obtain a 1-d global feature with PHASH
            # pil_img = Image.fromarray(gray)
            # phash = str(imagehash.phash(pil_img))
            # # Transfer hash value (hex) to decimal (PCA only takes in numerical values)
            # phash_dec = int(phash, 16)
            # sum_pash = sum_pash + phash_dec
            # item_count = item_count + 1

            # Train the index
            index.train(descriptors)
            # Add the descriptors to the index
            index.add(descriptors)
        else:
            print(f"Skipping non-image file: {filename}")

# Optimize the index for searching
index.nprobe = nprobe

# Search the index for the 5 nearest neighbors of a query vector
query = np.random.rand(1, local_feature_dimension).astype('float32')
distances, indices = index.search(query, k=5)

print('distances', distances)
print('indices', indices)

# Calculate the global feature
# group_global = sum_pash/item_count
# group_global_32 = struct.pack('f', group_global)
# print((group_global_32, type(group_global_32)))

# # Indexing based on local features
# local_feature_dimension = descriptors.shape[1]  # dimension of descriptors
# print(local_feature_dimension)
# nlist = 5 # 256  # number of clusters/centroids
# quantizer = faiss.IndexFlatL2(local_feature_dimension)  # the quantizer - L2 distance
# nsubquantizers = 8 # number of subquantizers
# nbits = 8 # number of bits per subquantizer
# index = faiss.IndexIVFPQ(quantizer, local_feature_dimension, nlist, nsubquantizers, nbits)
# # Train the index
# index.train(descriptors)
# # Add descriptors to the index with ids
# index.add_with_ids(descriptors, [i for i in range(len(descriptors))])
# # print('Add descriptors to the index with ids: ', index)
