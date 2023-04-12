# coding=utf-8
# Copyright 2018-2022 EVA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import unittest
from test.markers import windows_skip_marker
from test.util import file_remove, load_udfs_for_testing, shutdown_ray

import cv2
import numpy as np
import pytest

from eva.catalog.catalog_manager import CatalogManager
from eva.configuration.configuration_manager import ConfigurationManager
from eva.configuration.constants import EVA_ROOT_DIR
from eva.server.command_handler import execute_query_fetch_all
from eva.udfs.udf_bootstrap_queries import Asl_udf_query, Mvit_udf_query


@pytest.mark.notparallel
class PytorchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        CatalogManager().reset()
        ua_detrac = f"{EVA_ROOT_DIR}/data/ua_detrac/ua_detrac.mp4"
        mnist = f"{EVA_ROOT_DIR}/data/mnist/mnist.mp4"
        actions = f"{EVA_ROOT_DIR}/data/actions/actions.mp4"
        asl_actions = f"{EVA_ROOT_DIR}/data/actions/computer_asl.mp4"
        meme1 = f"{EVA_ROOT_DIR}/data/detoxify/meme1.jpg"
        meme2 = f"{EVA_ROOT_DIR}/data/detoxify/meme2.jpg"

        execute_query_fetch_all(f"LOAD VIDEO '{ua_detrac}' INTO MyVideo;")
        execute_query_fetch_all(f"LOAD VIDEO '{mnist}' INTO MNIST;")
        execute_query_fetch_all(f"LOAD VIDEO '{actions}' INTO Actions;")
        execute_query_fetch_all(f"LOAD VIDEO '{asl_actions}' INTO Asl_actions;")
        execute_query_fetch_all(f"LOAD IMAGE '{meme1}' INTO MemeImages;")
        execute_query_fetch_all(f"LOAD IMAGE '{meme2}' INTO MemeImages;")
        load_udfs_for_testing()

    @classmethod
    def tearDownClass(cls):
        shutdown_ray()

        file_remove("ua_detrac.mp4")
        file_remove("mnist.mp4")
        file_remove("actions.mp4")
        file_remove("computer_asl.mp4")

        execute_query_fetch_all("DROP TABLE IF EXISTS Actions;")
        execute_query_fetch_all("DROP TABLE IF EXISTS MNIST;")
        execute_query_fetch_all("DROP TABLE IF EXISTS MyVideo;")
        execute_query_fetch_all("DROP TABLE IF EXISTS Asl_actions;")
        execute_query_fetch_all("DROP TABLE IF EXISTS MemeImages;")

    @pytest.mark.torchtest
    def test_should_run_pytorch_and_fastrcnn_with_lateral_join(self):
        select_query = """SELECT id, obj.labels
                          FROM MyVideo JOIN LATERAL
                          FastRCNNObjectDetector(data)
                          AS obj(labels, bboxes, scores)
                         WHERE id < 2;"""
        actual_batch = execute_query_fetch_all(select_query)
        self.assertEqual(len(actual_batch), 2)

    @pytest.mark.torchtest
    def test_should_run_pytorch_and_yolo_and_mvit(self):
        execute_query_fetch_all(Mvit_udf_query)

        select_query = """SELECT FIRST(id),
                            YoloV5(FIRST(data)),
                            MVITActionRecognition(SEGMENT(data))
                            FROM Actions
                            WHERE id < 32
                            GROUP BY '16f'; """
        actual_batch = execute_query_fetch_all(select_query)
        self.assertEqual(len(actual_batch), 2)

        res = actual_batch.frames
        for idx in res.index:
            self.assertTrue(
                "person" in res["yolov5.labels"][idx]
                and "yoga" in res["mvitactionrecognition.labels"][idx]
            )

    @pytest.mark.torchtest
    def test_should_run_pytorch_and_asl(self):
        execute_query_fetch_all(Asl_udf_query)
        select_query = """SELECT FIRST(id), ASLActionRecognition(SEGMENT(data))
                        FROM Asl_actions
                        SAMPLE 5
                        GROUP BY '16f';"""
        actual_batch = execute_query_fetch_all(select_query)

        res = actual_batch.frames

        self.assertEqual(len(res), 1)
        for idx in res.index:
            self.assertTrue("computer" in res["aslactionrecognition.labels"][idx])

    @pytest.mark.torchtest
    def test_should_run_pytorch_and_yolo_decorators(self):
        create_udf_query = """CREATE UDF YoloDecorators
                  IMPL  'eva/udfs/decorators/yolo_object_detection_decorators.py';
        """
        execute_query_fetch_all(create_udf_query)

        select_query = """SELECT YoloDecorators(data) FROM MyVideo
                        WHERE id < 5;"""
        actual_batch = execute_query_fetch_all(select_query)
        self.assertEqual(len(actual_batch), 5)
    
    @pytest.mark.torchtest
    def test_xli3068(self):
        create_udf_query = """CREATE UDF IF NOT EXISTS ORB
                  INPUT  (frame NDARRAY UINT8(3, ANYDIM, ANYDIM))
                  OUTPUT (local_features NDARRAY FLOAT32(ANYDIM))
                  TYPE  FaceDetection
                  IMPL  'eva/udfs/local_feature.py';
        """
        execute_query_fetch_all(create_udf_query)

        select_query = """SELECT ORB(data) FROM MyVideo
                        WHERE id < 5;"""
        actual_batch = execute_query_fetch_all(select_query)
        self.assertEqual(len(actual_batch), 5)

    @pytest.mark.torchtest
    def test_should_run_pytorch_and_facenet(self):
        create_udf_query = """CREATE UDF FaceDetector
                  INPUT  (frame NDARRAY UINT8(3, ANYDIM, ANYDIM))
                  OUTPUT (bboxes NDARRAY FLOAT32(ANYDIM, 4),
                          scores NDARRAY FLOAT32(ANYDIM))
                  TYPE  FaceDetection
                  IMPL  'eva/udfs/face_detector.py';
        """
        execute_query_fetch_all(create_udf_query)

        select_query = """SELECT FaceDetector(data) FROM MyVideo
                        WHERE id < 5;"""
        actual_batch = execute_query_fetch_all(select_query)
        self.assertEqual(len(actual_batch), 5)

    @pytest.mark.torchtest
    @windows_skip_marker
    def test_should_run_pytorch_and_ocr(self):
        create_udf_query = """CREATE UDF IF NOT EXISTS OCRExtractor
                  INPUT  (frame NDARRAY UINT8(3, ANYDIM, ANYDIM))
                  OUTPUT (labels NDARRAY STR(10),
                          bboxes NDARRAY FLOAT32(ANYDIM, 4),
                          scores NDARRAY FLOAT32(ANYDIM))
                  TYPE  OCRExtraction
                  IMPL  'eva/udfs/ocr_extractor.py';
        """
        execute_query_fetch_all(create_udf_query)

        select_query = """SELECT OCRExtractor(data) FROM MNIST
                        WHERE id >= 150 AND id < 155;"""
        actual_batch = execute_query_fetch_all(select_query)
        self.assertEqual(len(actual_batch), 5)

        # non-trivial test case for MNIST
        res = actual_batch.frames
        self.assertTrue(res["ocrextractor.labels"][0][0] == "4")
        self.assertTrue(res["ocrextractor.scores"][2][0] > 0.9)

    @pytest.mark.torchtest
    def test_should_run_pytorch_and_resnet50(self):
        create_udf_query = """CREATE UDF IF NOT EXISTS FeatureExtractor
                  INPUT  (frame NDARRAY UINT8(3, ANYDIM, ANYDIM))
                  OUTPUT (features NDARRAY FLOAT32(ANYDIM))
                  TYPE  Classification
                  IMPL  'eva/udfs/feature_extractor.py';
        """
        execute_query_fetch_all(create_udf_query)

        select_query = """SELECT FeatureExtractor(data) FROM MyVideo
                        WHERE id < 5;"""
        actual_batch = execute_query_fetch_all(select_query)
        self.assertEqual(len(actual_batch), 5)

        # non-trivial test case for Resnet50
        res = actual_batch.frames
        self.assertEqual(res["featureextractor.features"][0].shape, (1, 2048))
        # self.assertTrue(res["featureextractor.features"][0][0][0] > 0.3)

    @pytest.mark.torchtest
    def test_should_run_pytorch_and_similarity(self):
        create_open_udf_query = """CREATE UDF IF NOT EXISTS Open
                INPUT (img_path TEXT(1000))
                OUTPUT (data NDARRAY UINT8(3, ANYDIM, ANYDIM))
                TYPE NdarrayUDF
                IMPL "eva/udfs/ndarray/open.py";
        """
        execute_query_fetch_all(create_open_udf_query)

        create_similarity_udf_query = """CREATE UDF IF NOT EXISTS Similarity
                    INPUT (Frame_Array_Open NDARRAY UINT8(3, ANYDIM, ANYDIM),
                           Frame_Array_Base NDARRAY UINT8(3, ANYDIM, ANYDIM),
                           Feature_Extractor_Name TEXT(100))
                    OUTPUT (distance FLOAT(32, 7))
                    TYPE NdarrayUDF
                    IMPL "eva/udfs/ndarray/similarity.py";
        """
        execute_query_fetch_all(create_similarity_udf_query)

        create_feat_udf_query = """CREATE UDF IF NOT EXISTS FeatureExtractor
                  INPUT  (frame NDARRAY UINT8(3, ANYDIM, ANYDIM))
                  OUTPUT (features NDARRAY FLOAT32(ANYDIM))
                  TYPE  Classification
                  IMPL  "eva/udfs/feature_extractor.py";
        """
        execute_query_fetch_all(create_feat_udf_query)

        select_query = """SELECT data FROM MyVideo WHERE id = 1;"""
        batch_res = execute_query_fetch_all(select_query)
        img = batch_res.frames["myvideo.data"][0]

        config = ConfigurationManager()
        tmp_dir_from_config = config.get_value("storage", "tmp_dir")

        img_save_path = os.path.join(tmp_dir_from_config, "dummy.jpg")
        try:
            os.remove(img_save_path)
        except FileNotFoundError:
            pass
        cv2.imwrite(img_save_path, img)

        similarity_query = """SELECT data FROM MyVideo WHERE id < 5
                    ORDER BY Similarity(FeatureExtractor(Open("{}")),
                                        FeatureExtractor(data))
                    LIMIT 1;""".format(
            img_save_path
        )
        actual_batch = execute_query_fetch_all(similarity_query)

        similar_data = actual_batch.frames["myvideo.data"][0]
        self.assertTrue(np.array_equal(img, similar_data))

    @pytest.mark.torchtest
    @windows_skip_marker
    def test_should_run_ocr_on_cropped_data(self):
        create_udf_query = """CREATE UDF IF NOT EXISTS OCRExtractor
                  INPUT  (text NDARRAY STR(100))
                  OUTPUT (labels NDARRAY STR(10),
                          bboxes NDARRAY FLOAT32(ANYDIM, 4),
                          scores NDARRAY FLOAT32(ANYDIM))
                  TYPE  OCRExtraction
                  IMPL  'eva/udfs/ocr_extractor.py';
        """
        execute_query_fetch_all(create_udf_query)

        select_query = """SELECT OCRExtractor(Crop(data, [2, 2, 24, 24])) FROM MNIST
                        WHERE id >= 150 AND id < 155;"""
        actual_batch = execute_query_fetch_all(select_query)
        self.assertEqual(len(actual_batch), 5)

        # non-trivial test case for MNIST
        res = actual_batch.frames
        self.assertTrue(res["ocrextractor.labels"][0][0] == "4")
        self.assertTrue(res["ocrextractor.scores"][2][0] > 0.9)

    @pytest.mark.torchtest
    @windows_skip_marker
    def test_should_run_detoxify_on_text(self):
        create_udf_query = """CREATE UDF IF NOT EXISTS OCRExtractor
                  INPUT  (text NDARRAY STR(100))
                  OUTPUT (labels NDARRAY STR(10),
                          bboxes NDARRAY FLOAT32(ANYDIM, 4),
                          scores NDARRAY FLOAT32(ANYDIM))
                  TYPE  OCRExtraction
                  IMPL  'eva/udfs/ocr_extractor.py';
        """
        execute_query_fetch_all(create_udf_query)

        create_udf_query = """CREATE UDF IF NOT EXISTS ToxicityClassifier
                  INPUT  (text NDARRAY STR(100))
                  OUTPUT (labels NDARRAY STR(10))
                  TYPE  Classification
                  IMPL  'eva/udfs/toxicity_classifier.py';
        """
        execute_query_fetch_all(create_udf_query)

        select_query = """SELECT name, OCRExtractor(data).labels,
                                 ToxicityClassifier(OCRExtractor(data).labels)
                        FROM MemeImages;"""
        actual_batch = execute_query_fetch_all(select_query)

        # non-trivial test case for Detoxify
        res = actual_batch.frames
        for i in range(2):
            # Image can be reordered.
            if "meme1" in res["memeimages.name"][i]:
                self.assertTrue(res["toxicityclassifier.labels"][i] == "toxic")
            else:
                self.assertTrue(res["toxicityclassifier.labels"][i] == "not toxic")

    def test_check_unnest_with_predicate_on_yolo(self):
        query = """SELECT id, yolov5.label, yolov5.bbox, yolov5.score
                  FROM MyVideo
                  JOIN LATERAL UNNEST(YoloV5(data)) AS yolov5(label, bbox, score)
                  WHERE yolov5.label = 'car' AND id < 10;"""

        actual_batch = execute_query_fetch_all(query)

        # due to unnest the number of returned tuples should be atleast > 10
        self.assertTrue(len(actual_batch) > 10)


# import unittest

# import pytest

# from eva.server.command_handler import execute_query_fetch_all

# @pytest
# def test_should_load_video():
#     select_query = """SELECT id FROM MyVideo WHERE id < 5;"""
#     actual_batch = execute_query_fetch_all(select_query)
#     assert len(actual_batch) == 5

# @pytest
# def test_should_create_UDF():
#     create_udf_query = """CREATE UDF IF NOT EXISTS ORB
#             INPUT  (frame NDARRAY UINT8(3, ANYDIM, ANYDIM))
#             OUTPUT (local_features NDARRAY FLOAT32(ANYDIM))
#             TYPE  FaceDetection
#             IMPL  'eva/udfs/local_feature.py';"""
#     execute_query_fetch_all(create_udf_query)

#     select_query = """SELECT ORB(data) FROM MyVideo
#                     WHERE id < 5;"""

#     actual_batch = execute_query_fetch_all(select_query)
#     assert len(actual_batch) == 5

# @pytest.mark.torchtest
# @pytest.mark.benchmark(
#     warmup=False,
#     warmup_iterations=1,
#     min_rounds=1,
# )
# @pytest.mark.notparallel
# def test_should_run_pytorch_and_yolo(benchmark, setup_pytorch_tests):
#     select_query = """SELECT YoloV5(data) FROM MyVideo
#                     WHERE id < 5;"""
#     actual_batch = benchmark(execute_query_fetch_all, select_query)
#     assert len(actual_batch) == 5


# @pytest.mark.torchtest
# @pytest.mark.benchmark(
#     warmup=False,
#     warmup_iterations=1,
#     min_rounds=1,
# )
# @pytest.mark.notparallel
# def test_should_run_pytorch_and_facenet(benchmark, setup_pytorch_tests):
#     create_udf_query = """CREATE UDF IF NOT EXISTS FaceDetector
#                 INPUT  (frame NDARRAY UINT8(3, ANYDIM, ANYDIM))
#                 OUTPUT (bboxes NDARRAY FLOAT32(ANYDIM, 4),
#                         scores NDARRAY FLOAT32(ANYDIM))
#                 TYPE  FaceDetection
#                 IMPL  'eva/udfs/face_detector.py';
#     """
#     execute_query_fetch_all(create_udf_query)

#     select_query = """SELECT FaceDetector(data) FROM MyVideo
#                     WHERE id < 5;"""

#     actual_batch = benchmark(execute_query_fetch_all, select_query)
#     assert len(actual_batch) == 5


# @pytest.mark.torchtest
# @pytest.mark.benchmark(
#     warmup=False,
#     warmup_iterations=1,
#     min_rounds=1,
# )
# @pytest.mark.notparallel
# def test_should_run_pytorch_and_resnet50(benchmark, setup_pytorch_tests):
#     create_udf_query = """CREATE UDF IF NOT EXISTS FeatureExtractor
#                 INPUT  (frame NDARRAY UINT8(3, ANYDIM, ANYDIM))
#                 OUTPUT (features NDARRAY FLOAT32(ANYDIM))
#                 TYPE  Classification
#                 IMPL  'eva/udfs/feature_extractor.py';
#     """
#     execute_query_fetch_all(create_udf_query)

#     select_query = """SELECT FeatureExtractor(data) FROM MyVideo
#                     WHERE id < 5;"""
#     actual_batch = benchmark(execute_query_fetch_all, select_query)
#     assert len(actual_batch) == 5

#     # non-trivial test case for Resnet50
#     res = actual_batch.frames
#     assert res["featureextractor.features"][0].shape == (1, 2048)
#     # assert res["featureextractor.features"][0][0][0] > 0.3


# @pytest.mark.torchtest
# @pytest.mark.benchmark(
#     warmup=False,
#     warmup_iterations=1,
#     min_rounds=1,
# )
# @pytest.mark.notparallel
# def test_lateral_join(benchmark, setup_pytorch_tests):
#     select_query = """SELECT id, a FROM MyVideo JOIN LATERAL
#                     YoloV5(data) AS T(a,b,c) WHERE id < 5;"""
#     actual_batch = benchmark(execute_query_fetch_all, select_query)
#     assert len(actual_batch) == 5
#     assert list(actual_batch.columns) == ["myvideo.id", "T.a"]

# import pytest

# class Calculator:
#     def add(self, a, b):
#         return a + b

#     def subtract(self, a, b):
#         return a - b

#     def multiply(self, a, b):
#         return a * b

#     def divide(self, a, b):
#         if b == 0:
#             raise ValueError("Cannot divide by zero")
#         return a / b

# def test_addition():
#     calculator = Calculator()
#     assert calculator.add(2, 3) == 5

# def test_subtraction():
#     calculator = Calculator()
#     assert calculator.subtract(5, 2) == 3

# def test_multiplication():
#     calculator = Calculator()
#     assert calculator.multiply(2, 4) == 8

# def test_division():
#     calculator = Calculator()
#     assert calculator.divide(10, 2) == 5

# def test_divide_by_zero_raises_error():
#     calculator = Calculator()
#     with pytest.raises(ValueError):
#         calculator.divide(10, 0)

# if __name__ == "__main__":
#     pytest.main()



# # coding=utf-8
# # Copyright 2018-2022 EVA
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# import unittest
# from test.util import create_sample_image, load_udfs_for_testing, shutdown_ray

# import numpy as np
# import pandas as pd
# import pytest

# from eva.catalog.catalog_manager import CatalogManager
# from eva.models.storage.batch import Batch
# from eva.server.command_handler import execute_query_fetch_all
# from eva.storage.storage_engine import StorageEngine


# @pytest.mark.notparallel
# class SimilarityTests(unittest.TestCase):
#     def setUp(self):
#         CatalogManager().reset()

#         # Prepare needed UDFs and data_col.
#         load_udfs_for_testing(mode="minimal")
#         self.img_path = create_sample_image()

#         # Create base comparison table.
#         create_table_query = """CREATE TABLE IF NOT EXISTS testSimilarityTable
#                                   (data_col NDARRAY UINT8(3, ANYDIM, ANYDIM),
#                                    dummy INTEGER);"""
#         execute_query_fetch_all(create_table_query)

#         # Create feature table.
#         create_table_query = """CREATE TABLE IF NOT EXISTS testSimilarityFeatureTable
#                                   (feature_col NDARRAY FLOAT32(1, ANYDIM),
#                                    dummy INTEGER);"""
#         execute_query_fetch_all(create_table_query)

#         # Prepare injected data_col.
#         base_img = np.array(np.ones((3, 3, 3)), dtype=np.uint8)
#         base_img[0] -= 1
#         base_img[2] += 1

#         # id: 1 -> most dissimilar, id: 5 -> most similar
#         base_img += 4

#         # Inject data_col.
#         base_table_catalog_entry = CatalogManager().get_table_catalog_entry(
#             "testSimilarityTable"
#         )
#         feature_table_catalog_entry = CatalogManager().get_table_catalog_entry(
#             "testSimilarityFeatureTable"
#         )
#         storage_engine = StorageEngine.factory(base_table_catalog_entry)
#         for i in range(5):
#             storage_engine.write(
#                 base_table_catalog_entry,
#                 Batch(
#                     pd.DataFrame(
#                         [
#                             {
#                                 "data_col": base_img,
#                                 "dummy": i,
#                             }
#                         ]
#                     )
#                 ),
#             )
#             storage_engine.write(
#                 feature_table_catalog_entry,
#                 Batch(
#                     pd.DataFrame(
#                         [
#                             {
#                                 "feature_col": base_img.astype(np.float32).reshape(
#                                     1, -1
#                                 ),
#                                 "dummy": i,
#                             }
#                         ]
#                     )
#                 ),
#             )
#             base_img -= 1

#     def tearDown(self):
#         shutdown_ray()

#         drop_table_query = "DROP TABLE testSimilarityTable;"
#         execute_query_fetch_all(drop_table_query)
#         drop_table_query = "DROP TABLE testSimilarityFeatureTable;"
#         execute_query_fetch_all(drop_table_query)

#     def test_similarity_should_work_in_order(self):
#         ###############################################
#         # Test case runs with UDF on raw input table. #
#         ###############################################

#         # Top 1 - assume table contains base data_col.
#         select_query = """SELECT data_col FROM testSimilarityTable
#                             ORDER BY Similarity(DummyFeatureExtractor(Open("{}")), DummyFeatureExtractor(data_col))
#                             LIMIT 1;""".format(
#             self.img_path
#         )
#         actual_batch = execute_query_fetch_all(select_query)

#         base_img = np.array(np.ones((3, 3, 3)), dtype=np.uint8)
#         base_img[0] -= 1
#         base_img[2] += 1

#         actual_open = actual_batch.frames["testsimilaritytable.data_col"].to_numpy()[0]
#         self.assertTrue(np.array_equal(actual_open, base_img))
#         # actual_distance = actual_batch.frames["similarity.distance"].to_numpy()[0]
#         # self.assertEqual(actual_distance, 0)

#         # Top 2 - assume table contains base data.
#         select_query = """SELECT data_col FROM testSimilarityTable
#                             ORDER BY Similarity(DummyFeatureExtractor(Open("{}")), DummyFeatureExtractor(data_col))
#                             LIMIT 2;""".format(
#             self.img_path
#         )
#         actual_batch = execute_query_fetch_all(select_query)

#         actual_open = actual_batch.frames["testsimilaritytable.data_col"].to_numpy()[0]
#         self.assertTrue(np.array_equal(actual_open, base_img))
#         actual_open = actual_batch.frames["testsimilaritytable.data_col"].to_numpy()[1]
#         self.assertTrue(np.array_equal(actual_open, base_img + 1))
#         # actual_distance = actual_batch.frames["similarity.distance"].to_numpy()[0]
#         # self.assertEqual(actual_distance, 0)
#         # actual_distance = actual_batch.frames["similarity.distance"].to_numpy()[1]
#         # self.assertEqual(actual_distance, 27)

#         ###########################################
#         # Test case runs on feature vector table. #
#         ###########################################

#         # Top 1 - assume table contains feature data.
#         select_query = """SELECT feature_col FROM testSimilarityFeatureTable
#                             ORDER BY Similarity(DummyFeatureExtractor(Open("{}")), feature_col)
#                             LIMIT 1;""".format(
#             self.img_path
#         )
#         actual_batch = execute_query_fetch_all(select_query)

#         base_img = np.array(np.ones((3, 3, 3)), dtype=np.uint8)
#         base_img[0] -= 1
#         base_img[2] += 1
#         base_img = base_img.astype(np.float32).reshape(1, -1)

#         actual_open = actual_batch.frames[
#             "testsimilarityfeaturetable.feature_col"
#         ].to_numpy()[0]
#         self.assertTrue(np.array_equal(actual_open, base_img))
#         # actual_distance = actual_batch.frames["similarity.distance"].to_numpy()[0]
#         # self.assertEqual(actual_distance, 0)

#         # Top 2 - assume table contains feature data.
#         select_query = """SELECT feature_col FROM testSimilarityFeatureTable
#                             ORDER BY Similarity(DummyFeatureExtractor(Open("{}")), feature_col)
#                             LIMIT 2;""".format(
#             self.img_path
#         )
#         actual_batch = execute_query_fetch_all(select_query)

#         actual_open = actual_batch.frames[
#             "testsimilarityfeaturetable.feature_col"
#         ].to_numpy()[0]
#         self.assertTrue(np.array_equal(actual_open, base_img))
#         actual_open = actual_batch.frames[
#             "testsimilarityfeaturetable.feature_col"
#         ].to_numpy()[1]
#         self.assertTrue(np.array_equal(actual_open, base_img + 1))
#         # actual_distance = actual_batch.frames["similarity.distance"].to_numpy()[0]
#         # self.assertEqual(actual_distance, 0)
#         # actual_distance = actual_batch.frames["similarity.distance"].to_numpy()[1]
#         # self.assertEqual(actual_distance, 27)

#     def test_should_do_faiss_index_scan(self):
#         ###########################################
#         # Test case runs on feature vector table. #
#         ###########################################

#         # Execution without index scan.
#         select_query = """SELECT feature_col FROM testSimilarityFeatureTable
#                             ORDER BY Similarity(DummyFeatureExtractor(Open("{}")), feature_col)
#                             LIMIT 3;""".format(
#             self.img_path
#         )
#         expected_batch = execute_query_fetch_all(select_query)

#         # Execution with index scan.
#         create_index_query = """CREATE INDEX testFaissIndexScanRewrite1
#                                     ON testSimilarityFeatureTable (feature_col)
#                                     USING HNSW;"""
#         execute_query_fetch_all(create_index_query)
#         select_query = """SELECT feature_col FROM testSimilarityFeatureTable
#                             ORDER BY Similarity(DummyFeatureExtractor(Open("{}")), feature_col)
#                             LIMIT 3;""".format(
#             self.img_path
#         )
#         explain_query = """EXPLAIN {}""".format(select_query)
#         explain_batch = execute_query_fetch_all(explain_query)
#         self.assertTrue("FaissIndexScan" in explain_batch.frames[0][0])
#         actual_batch = execute_query_fetch_all(select_query)

#         self.assertEqual(len(actual_batch), 3)
#         for i in range(3):
#             self.assertTrue(
#                 np.array_equal(
#                     expected_batch.frames[
#                         "testsimilarityfeaturetable.feature_col"
#                     ].to_numpy()[i],
#                     actual_batch.frames[
#                         "testsimilarityfeaturetable.feature_col"
#                     ].to_numpy()[i],
#                 )
#             )

#         ###############################################
#         # Test case runs with UDF on raw input table. #
#         ###############################################

#         # Execution without index scan.
#         select_query = """SELECT data_col FROM testSimilarityTable
#                             ORDER BY Similarity(DummyFeatureExtractor(Open("{}")), DummyFeatureExtractor(data_col))
#                             LIMIT 3;""".format(
#             self.img_path
#         )
#         expected_batch = execute_query_fetch_all(select_query)

#         # Execution with index scan.
#         create_index_query = """CREATE INDEX testFaissIndexScanRewrite2
#                                     ON testSimilarityTable (DummyFeatureExtractor(data_col))
#                                     USING HNSW;"""
#         execute_query_fetch_all(create_index_query)
#         select_query = """SELECT data_col FROM testSimilarityTable
#                             ORDER BY Similarity(DummyFeatureExtractor(Open("{}")), DummyFeatureExtractor(data_col))
#                             LIMIT 3;""".format(
#             self.img_path
#         )
#         explain_query = """EXPLAIN {}""".format(select_query)
#         explain_batch = execute_query_fetch_all(explain_query)
#         self.assertTrue("FaissIndexScan" in explain_batch.frames[0][0])
#         actual_batch = execute_query_fetch_all(select_query)

#         self.assertEqual(len(actual_batch), 3)
#         for i in range(3):
#             self.assertTrue(
#                 np.array_equal(
#                     expected_batch.frames["testsimilaritytable.data_col"].to_numpy()[i],
#                     actual_batch.frames["testsimilaritytable.data_col"].to_numpy()[i],
#                 )
#             )

#         # Cleanup
#         CatalogManager().drop_index_catalog_entry("testFaissIndexScanRewrite1")
#         CatalogManager().drop_index_catalog_entry("testFaissIndexScanRewrite2")

#     def test_should_not_do_faiss_index_scan_with_predicate(self):
#         # Execution with index scan.
#         create_index_query = """CREATE INDEX testFaissIndexScanRewrite
#                                     ON testSimilarityTable (DummyFeatureExtractor(data_col))
#                                     USING HNSW;"""
#         execute_query_fetch_all(create_index_query)

#         explain_query = """
#             EXPLAIN
#                 SELECT data_col FROM testSimilarityTable WHERE dummy = 0
#                   ORDER BY Similarity(DummyFeatureExtractor(Open("{}")), DummyFeatureExtractor(data_col))
#                   LIMIT 3;
#         """.format(
#             "dummypath"
#         )
#         batch = execute_query_fetch_all(explain_query)

#         # Index scan should not be used.
#         self.assertFalse("FaissIndexScan" in batch.frames[0][0])

#         # Cleanup
#         CatalogManager().drop_index_catalog_entry("testFaissIndexScanRewrite")
