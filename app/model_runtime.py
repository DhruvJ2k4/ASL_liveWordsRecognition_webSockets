# # -*- coding: utf-8 -*-
# # Python 3.7
# import os
# import numpy as np
# from typing import Dict, Any, Tuple, List
# from concurrent.futures import ThreadPoolExecutor

# # TensorFlow 2.3.1 / Keras API per your env
# from tensorflow.keras.models import load_model  # noqa: E402

# from .config import DIM, FRAMES, CHANNELS, MODEL_PATH, THRESHOLD, USE_DUMMY_MODEL
# from .labels import LABELS

# # A simple thread pool for offloading TF predict without blocking the event loop
# _EXECUTOR = ThreadPoolExecutor(max_workers=int(os.getenv("INFER_WORKERS", "2")))

# class DummyModel(object):
#     """Deterministic stand-in when a real .h5 isn't present (for tests)."""
#     def predict(self, x: np.ndarray) -> np.ndarray:
#         # x shape: (1, FRAMES, H, W, C)
#         batch = x.shape[0]
#         logits = np.zeros((batch, len(LABELS)), dtype="float32")
#         logits[:, 10] = 0.9  # push 'hello' as the default class
#         return logits

# def load_runtime_model():
#     if USE_DUMMY_MODEL:
#         return DummyModel()
#     if not os.path.exists(MODEL_PATH):
#         # Fallback gracefully (dev/test ergonomics).
#         return DummyModel()
#     return load_model(MODEL_PATH)

# class VideoInferenceService(object):
#     """Maintains a sliding window per connection and runs predictions safely."""
#     def __init__(self, model):
#         self.model = model
#         # buffer: (n, H, W, C) in RGB [0,1]
#         self._buffer = np.empty((0, DIM[1], DIM[0], CHANNELS), dtype="float32")

#     def add_frame(self, frame_rgb: np.ndarray) -> None:
#         frame_rgb = np.reshape(frame_rgb, (1, frame_rgb.shape[0], frame_rgb.shape[1], frame_rgb.shape[2]))
#         self._buffer = np.append(self._buffer, frame_rgb, axis=0)
#         # keep last FRAMES
#         if self._buffer.shape[0] > FRAMES:
#             self._buffer = self._buffer[-FRAMES:, :, :, :]

#     def ready(self) -> bool:
#         return self._buffer.shape[0] == FRAMES

#     def _predict_sync(self) -> Dict[str, Any]:
#         # model expects (1, T, H, W, C)
#         inp = self._buffer.reshape(1, FRAMES, DIM[1], DIM[0], CHANNELS)
#         probs = self.model.predict(inp)[0]  # shape: (num_classes,)
#         best_idx = int(np.argmax(probs))
#         best_prob = float(probs[best_idx])

#         if best_prob >= THRESHOLD:
#             label = LABELS.get(best_idx, str(best_idx))
#         else:
#             label = "none"

#         # Top-5 for UI/debugging
#         topk_idx = np.argsort(-probs)[:5].tolist()
#         topk = [{"index": int(i), "label": LABELS.get(int(i), str(i)), "score": float(probs[int(i)])}
#                 for i in topk_idx]

#         return {
#             "label": label,
#             "score": best_prob,
#             "index": best_idx,
#             "topk": topk,
#             "threshold": THRESHOLD,
#         }

#     async def predict(self) -> Dict[str, Any]:
#         loop = None
#         try:
#             import asyncio
#             loop = asyncio.get_running_loop()
#         except Exception:
#             pass
#         if loop is not None:
#             return await loop.run_in_executor(_EXECUTOR, self._predict_sync)
#         # Fallback (non-async context)
#         return self._predict_sync()
# -*- coding: utf-8 -*-
# Python 3.7
import os
import threading
import numpy as np
from typing import Dict, Any
from tensorflow.keras.models import load_model

from .config import DIM, FRAMES, CHANNELS, MODEL_PATH, THRESHOLD, USE_DUMMY_MODEL
from .labels import LABELS

# class DummyModel(object):
#     def predict(self, x: np.ndarray) -> np.ndarray:
#         # x: (1, T, H, W, C)
#         batch = x.shape[0]
#         num_classes = len(LABELS)
#         logits = np.zeros((batch, num_classes), dtype="float32")
#         logits[:, 10] = 0.9  # "hello"
#         return logits

# def load_runtime_model():
#     if USE_DUMMY_MODEL:
#         return DummyModel()
#     if not os.path.exists(MODEL_PATH):
#         return DummyModel()
#     return load_model(MODEL_PATH)

_MODEL_SINGLETON = None

class DummyModel(object):
    def predict(self, x: np.ndarray) -> np.ndarray:
        batch = x.shape[0]
        logits = np.zeros((batch, len(LABELS)), dtype="float32")
        logits[:, 10] = 0.9
        return logits

def load_runtime_model():
    global _MODEL_SINGLETON
    if _MODEL_SINGLETON is not None:
        return _MODEL_SINGLETON

    if USE_DUMMY_MODEL:
        _MODEL_SINGLETON = DummyModel()
        return _MODEL_SINGLETON

    if not os.path.exists(MODEL_PATH):
        _MODEL_SINGLETON = DummyModel()
        return _MODEL_SINGLETON

    _MODEL_SINGLETON = load_model(MODEL_PATH)
    return _MODEL_SINGLETON

class VideoInferenceService(object):
    """
    EXACT behavior parity with the user's script for:
      - buffer management
      - reshape order
      - argmax + threshold
    One active prediction thread at a time (like 'x' in the script).
    """
    def __init__(self, model):
        self.model = model
        # buffer: (n, H, W, C)  -> n grows up to FRAMES
        self._buffer = np.empty((0, DIM[1], DIM[0], CHANNELS), dtype="float32")
        self._worker = threading.Thread()
        self._last_result = {"label": "none", "score": 0.0}

    def add_frame_already_normalized(self, frame_norm: np.ndarray) -> None:
        """
        Append a single normalized frame (H, W, C) exactly as user's code:
          frame_resh = np.reshape(frame_res, (1, H, W, C))
          np.append(buffer, frame_resh, axis=0)
        """
        frame_resh = np.reshape(frame_norm, (1, frame_norm.shape[0], frame_norm.shape[1], frame_norm.shape[2]))
        self._buffer = np.append(self._buffer, frame_resh, axis=0)

        # left-shift when over capacity (keep semantics clean)
        if self._buffer.shape[0] > FRAMES:
            self._buffer = self._buffer[1:FRAMES, :, :, :]

    def ready(self) -> bool:
        return self._buffer.shape[0] == FRAMES

    def _predict_sync(self):
        # EXACT reshape: frame_buffer.reshape(1, *frame_buffer.shape)
        inp = self._buffer.reshape(1, *self._buffer.shape)  # (1, T, H, W, C)
        preds = self.model.predict(inp)[0]                  # (num_classes,)
        best_idx = int(np.argmax(preds))
        best_prob = float(preds[best_idx])

        if best_prob > THRESHOLD:
            label = LABELS.get(best_idx, str(best_idx))
        else:
            label = "none"

        self._last_result = {"label": label, "score": best_prob}
        return self._last_result

    def maybe_launch_prediction(self) -> bool:
        """
        Mirror: if not x.is_alive(): x = Thread(target=make_prediction, ...); x.start()
        Returns True if a new prediction was started, False otherwise.
        """
        if not self.ready():
            return False
        if not self._worker.is_alive():
            self._worker = threading.Thread(target=self._predict_sync, daemon=True)
            self._worker.start()
            return True
        return False

    def current_result(self) -> Dict[str, Any]:
        return dict(self._last_result)
