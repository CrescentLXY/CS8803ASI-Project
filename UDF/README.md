# UDF

The current pipeline can be directly realized with built-in EVA functions:

xli3068.py: a self-written UDF to extract the features and perform transformation to fit in INDEX function. Mostly based on the original FeatureExtractor UDF in EVA.

Place under eva/udfs/

Python-pipeline: The image similarity analysis pipeline implemented in Python. Randomly select a query, find and show the two most similar images.

'''

Previous trials:

motif-mining.py: Script to extract and index features (apply PCA directly on local features to extract a global feature, and indexing is performed based only on local features)

loca-feauture.py: UDF to extract only local features (testing purpose, functions to indexing and IVF generation are commented out)

'''
