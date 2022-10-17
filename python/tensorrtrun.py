"""onnx图片需要调整BGR2RGB, 并调整通道为[B, C, H, W], 且需要归一化
"""

from pathlib import Path

from typing import Sequence

import numpy as np
import cv2
import time
import torch

from engine import TRTWrapper
from utils import resize_and_pad, post, get_index2label

import sys
import os
os.chdir(sys.path[0])


CONFIDENCE_THRESHOLD = 0.25
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.45


def get_image(image_path):
    """获取图像

    Args:
        image_path (str): 图片路径

    Returns:
        Tuple: 原图, 输入的tensor, 填充的宽, 填充的高
    """
    img = cv2.imread(str(Path(image_path)))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR2RGB

    img_reized, delta_w ,delta_h = resize_and_pad(img_rgb, (640, 640))

    img_reized = img_reized.astype(np.float32)
    img_reized /= 255.0                             # 归一化

    img_reized = img_reized.transpose(2, 0, 1)      # [H, W, C] -> [C, H, W]
    input_tensor = np.expand_dims(img_reized, 0)    # [C, H, W] -> [B, C, H, W]
    input_tensor = torch.from_numpy(input_tensor)

    return img, input_tensor, delta_w ,delta_h


def get_engine_model(engine_path, output_names: Sequence[str] = ['output0']):
    """获取模型

    Args:
        onnx_path (str): 模型路径

    Returns:
        InferenceSession: 推理模型
    """
    model = TRTWrapper(engine_path, output_names)
    return model


#--------------------------------#
#   推理
#--------------------------------#
def inference():
    ENGINE_PATH  = "../weights/yolov5s.engine"
    IMAGE_PATH = "../images/bus.jpg"
    YAML_PATH  = "../weights/yolov5s.yaml"

    # 获取图片,扩展的宽高
    img, input_tensor, delta_w ,delta_h = get_image(IMAGE_PATH)

    # 获取模型
    model = get_engine_model(ENGINE_PATH, ['output0'])

    # 获取label
    index2label = get_index2label(YAML_PATH)

    start = time.time()
    detections = model({"images": input_tensor.cuda()})
    # print(detections[0].shape)                                    # [1, 25200, 85]
    detections = np.squeeze(detections['output0'].cpu().numpy())    # [25200, 85]

    # Step 8. Postprocessing including NMS
    img = post(detections, delta_w ,delta_h, img, CONFIDENCE_THRESHOLD, SCORE_THRESHOLD, NMS_THRESHOLD, index2label)
    end = time.time()
    print((end - start) * 1000)

    cv2.imwrite("./engine_det.png", img)


if __name__ == "__main__":
    inference()
