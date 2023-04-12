from typing import List
import cv2 # use to load image, and use ORB to extract local feature
import pandas as pd
import numpy as np

from eva.udfs.abstract.abstract_udf import AbstractClassifierUDF

class ORB(AbstractClassifierUDF):
    def setup(self):
        self.orb = cv2.ORB_create(nfeatures=500)
    
    @property
    def name(self) -> str:
        return "ORB"
    
    @property
    def labels(self) -> List[str]:
        return []
    
    def forward(self, frames: pd.DataFrame) -> pd.DataFrame:
        """
        Performs predictions on input frames
        Arguments:
            frames (np.ndarray): Frames on which predictions need to be performed
        Returns:
            local features (List[local features])
        """

        frames_list = frames.transpose().values.tolist()[0]
        frames = np.asarray(frames_list)
        keypoints, descriptors = self.orb.detectAndCompute(frames, None)
        outcome = []
        for descriptor in descriptors:
            local_feature = []
            if descriptor is not None:
                local_feature = descriptor
            outcome.append({"local_features": local_feature})
        return pd.DataFrame(outcome, columns=["local_feature"])
        # for frame_boxes, frame_scores in zip(boxes, scores):
        #     pred_boxes = []
        #     pred_scores = []
        #     if frame_boxes is not None and frame_scores is not None:
        #         if not np.isnan(pred_boxes):
        #             pred_boxes = np.asarray(frame_boxes, dtype="int")
        #             pred_scores = frame_scores
        #         else:
        #             logger.warn(f"Nan entry in box {frame_boxes}")
        #     outcome.append(
        #         {"bboxes": pred_boxes, "scores": pred_scores},
        #     )

        # return pd.DataFrame(outcome, columns=["bboxes", "scores"])
