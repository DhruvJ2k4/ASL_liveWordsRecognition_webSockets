# -*- coding: utf-8 -*-
# Python 3.7
import base64
import cv2
import numpy as np
from typing import Tuple

def b64_to_bgr_image(b64_string: str) -> np.ndarray:
    """Decode a base64-encoded image into an OpenCV BGR ndarray."""
    data = base64.b64decode(b64_string)
    img_array = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise ValueError("Could not decode image from base64 payload")
    return img

def preprocess_frame_bgr_exact(img_bgr: np.ndarray, dim: Tuple[int, int]) -> np.ndarray:
    """
    EXACT match to user's script:
      frame_res = cv2.resize(frame, dim)
      frame_res = frame_res / 255.0
    No color conversion. Output shape: (H, W, 3), dtype float32.
    """
    frame_res = cv2.resize(img_bgr, dim)
    frame_res = frame_res.astype("float32") / 255.0
    return frame_res
