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

from typing import List

import pandas as pd
import torch
from torch import Tensor
from torchvision import models
import numpy as np
import cv2 # use to load image, and use ORB to extract local feature
import kornia

from eva.udfs.abstract.pytorch_abstract_udf import PytorchAbstractClassifierUDF


class xli3068(PytorchAbstractClassifierUDF):
    """ """

    # def setup(self):
    #     self.orb = cv2.ORB_create()
        # for param in self.model.parameters():
        #     param.requires_grad = False
        # self.model.fc = torch.nn.Identity()
        # self.model.eval()

    def setup(self):
        self.model = models.resnet50(weights="IMAGENET1K_V2", progress=False)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = torch.nn.Identity()
        self.model.eval()

    # def setup(self):
    #     self.model = kornia.feature.SIFTDescriptor(100)

    @property
    def name(self) -> str:
        return "xli3068"

    @property
    def labels(self) -> List[str]:
        return []
    
    # def forward(self, df: pd.DataFrame) -> pd.DataFrame:
    #     def _forward(row: pd.Series) -> np.ndarray:
    #         rgb_img = row[0]
    #         gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    #         resized_gray_img = cv2.resize(gray_img, (100, 100), interpolation=cv2.INTER_AREA)
    #         resized_gray_img = np.moveaxis(resized_gray_img, -1, 0)
    #         batch_resized_gray_img = np.expand_dims(resized_gray_img, axis=0)
    #         batch_resized_gray_img = np.expand_dims(batch_resized_gray_img, axis=0)
    #         batch_resized_gray_img = batch_resized_gray_img.astype(np.float32)

    #         with torch.no_grad():
    #             # keypoint, torch_features = self.orb.detectAndCompute(batch_resized_gray_img, None)
    #             torch_features = self.model(torch.from_numpy(batch_resized_gray_img))
    #             features = torch_features.numpy()

    #         features = features.reshape(1, -1)
    #         return pd.DataFrame({"features": [features]})
    #         return features
    
    #     ret = pd.DataFrame()
    #     ret["features"] = df.apply(_forward, axis=1)
    #     ret = pd.concat(df.apply(_forward, axis=1).tolist(), ignore_index=True)

    
        # return ret

    def forward(self, frames: Tensor) -> pd.DataFrame:
        """
        Performs feature extraction on input frames
        Arguments:
            frames (np.ndarray): Frames on which predictions need
            to be performed

        Returns:
            features (List[float]) in format of NDARRAY FLOAT32(1, ANYDIM)
        """
        outcome = []
        for f in frames:
            with torch.no_grad():
                outcome.append(
                    {"features": self.as_numpy(self.model(torch.unsqueeze(f, 0))).reshape(1, -1)},
                )
            # with torch.no_grad():
            #     output = self.model(torch.unsqueeze(f, 0))
            #     features = output.detach().numpy().astype(np.float32)
            #     features = features.reshape((1, -1))
            #     outcome.append({"features": features})
                # outcome.append(features)
        return pd.DataFrame(outcome, columns=["features"])

# CREATE UDF IF NOT EXISTS xli3068 INPUT (frame NDARRAY UINT8(3, ANYDIM, ANYDIM)) OUTPUT (features NDARRAY FLOAT32(1, ANYDIM)) TYPE Classificaiton IMPL "eva/udfs/xli3068.py";
