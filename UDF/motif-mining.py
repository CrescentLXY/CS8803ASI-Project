from typing import List
import torch # check for device
import cv2 # use to load image, and use ORB to extract local feature
from sklearn.decomposition import PCA # use PCA to reduce dimensions of global feature
import faiss # use FAISS for indexing
import pandas as pd
import numpy as np

from eva.udfs.abstract.abstract_udf import AbstractClassifierUDF

class MotifMining(AbstractClassifierUDF):
    def setup(self):
        self.orb = cv2.ORB_create(nfeatures=9984)
        self.pca = PCA(n_components=16)
        self.d = 32  # dimension of descriptors ?
        self.nlist = 256  # number of clusters/centroids
        self.nsubquantizers = 8  # number of subquantizers
        self.nbits = 8  # number of bits per subquantizer
        self.quantizer = faiss.IndexFlatL2(self.d)  # the quantizer - L2 distance
        self.index = faiss.IndexIVFPQ(self.quantizer, self.d, self.nlist, self.nsubquantizers, self.nbits)
        
    @property
    def name(self):
        return "MotifMining"
    
    @property
    def labels(self) -> List[str]:
        return []

    def forward(self, frames: pd.DataFrame) -> pd.DataFrame:
        """
        Performs feature extraction and indexing on input frames
        Arguments:
            frames (np.ndarray): Frames on which operations need to be performed
        Returns:
            overall feature (List[List[features]])
        """

        frames_list = frames.transpose().values.tolist()[0]
        frames = np.asarray(frames_list)

        # Assuming 'frames' is a pandas dataframe with image data
        # Extract the images from the dataframe and convert to grayscale
        images = []
        for frame in frames.values:
            image_data = np.frombuffer(frame[0], np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)
            images.append(image)
        
        # Perform local and global feature extraction on the images
        orb = self.orb
        # keypoints = []
        # descriptors = []
        local_features = []
        global_features = []
        for image in images:
            kp, des = orb.detectAndCompute(image, None)
            if des is not None:
                local_features.append(des.tolist())
                pca_features = self.pca.fit_transform(des)
                global_features.append(pca_features.tolist())
        
        return pd.DataFrame({"local_features": local_features, "global_features": global_features})
        
        # # Concatenate descriptors for each image
        # descriptors_concat = []
        # for des in descriptors:
        #     if des is not None:
        #         descriptors_concat.append(des)
        # descriptors_concat = np.concatenate(descriptors_concat, axis=0)

        # # Compute PCA on concatenated descriptors
        # pca_features = self.pca.fit_transform(descriptors_concat)
        
        # # Train the index
        # self.index.train(pca_features)

        # # Add global features of new images to the index
        # for i in range(len(pca_features)):
        #     self.index.add(pca_features[i:i+1, :])

        # # Query the index to find similar images
        # D, I = self.index.search(pca_features, k=10)

        # # create Pandas DataFrame with columns 'D' and 'I'
        # result_df = pd.DataFrame({'D': D.tolist(), 'I': I.tolist()})
        # return result_df
