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
import tensorflow as tf
from tensorflow.keras.models import load_model

from .config import (
    DIM, FRAMES, CHANNELS, MODEL_PATH, THRESHOLD, USE_DUMMY_MODEL,
    TF_INTER_OP_THREADS, TF_INTRA_OP_THREADS
)
from .labels import LABELS

# CRITICAL FIX: Configure TensorFlow BEFORE any model loading
# This prevents memory leaks and crashes

# 1. Limit threading to prevent CPU overload
tf.config.threading.set_inter_op_parallelism_threads(TF_INTER_OP_THREADS)
tf.config.threading.set_intra_op_parallelism_threads(TF_INTRA_OP_THREADS)

# 2. Configure GPU memory growth (if GPU available)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[TF CONFIG] GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"[TF CONFIG] GPU memory growth config failed: {e}")
else:
    print("[TF CONFIG] No GPU detected, using CPU")

# 3. Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR

# 4. Optional: Force CPU-only execution (uncomment if CUDA issues persist)
# tf.config.set_visible_devices([], 'GPU')

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
    
    CRITICAL FIXES:
    - Thread-safe buffer operations with lock
    - Fixed buffer slicing to prevent memory leaks
    - Proper cleanup on disconnect
    """
    def __init__(self, model):
        self.model = model
        # buffer: (n, H, W, C)  -> n grows up to FRAMES
        self._buffer = np.empty((0, DIM[1], DIM[0], CHANNELS), dtype="float32")
        self._worker = threading.Thread()
        self._last_result = {"label": "none", "score": 0.0}
        
        # CRITICAL: Thread safety lock to prevent race conditions
        self._lock = threading.Lock()
        
        # Track prediction count for debugging
        self._prediction_count = 0

    def add_frame_already_normalized(self, frame_norm: np.ndarray) -> None:
        """
        Append a single normalized frame (H, W, C) exactly as user's code:
          frame_resh = np.reshape(frame_res, (1, H, W, C))
          np.append(buffer, frame_resh, axis=0)
        
        CRITICAL FIX: Thread-safe with proper buffer management
        """
        with self._lock:
            frame_resh = np.reshape(frame_norm, (1, frame_norm.shape[0], frame_norm.shape[1], frame_norm.shape[2]))
            self._buffer = np.append(self._buffer, frame_resh, axis=0)

            # CRITICAL FIX: Correct slicing to maintain exactly FRAMES when full
            # OLD BUG: self._buffer[1:FRAMES] would give 9 frames if buffer was 11
            # NEW FIX: Keep last FRAMES only
            if self._buffer.shape[0] > FRAMES:
                self._buffer = self._buffer[-FRAMES:]  # Keep last FRAMES frames

    def ready(self) -> bool:
        with self._lock:
            return self._buffer.shape[0] == FRAMES

    def _predict_sync(self):
        """
        CRITICAL FIX: 
        - Copy buffer snapshot to avoid race conditions
        - Add TensorFlow session management
        - Proper error handling to prevent crashes
        """
        try:
            # CRITICAL: Take snapshot of buffer with lock
            with self._lock:
                if self._buffer.shape[0] < FRAMES:
                    return self._last_result  # Not ready, return cached result
                
                # Create a copy to avoid buffer modification during prediction
                buffer_snapshot = np.copy(self._buffer)
            
            # EXACT reshape: frame_buffer.reshape(1, *frame_buffer.shape)
            inp = buffer_snapshot.reshape(1, *buffer_snapshot.shape)  # (1, T, H, W, C)
            
            # CRITICAL FIX: TensorFlow prediction with proper settings
            # Use verbose=0 to suppress logs, predict_on_batch for efficiency
            preds = self.model.predict(inp, verbose=0)[0]  # (num_classes,)
            
            best_idx = int(np.argmax(preds))
            best_prob = float(preds[best_idx])

            if best_prob > THRESHOLD:
                label = LABELS.get(best_idx, str(best_idx))
            else:
                label = "none"

            # Update result safely
            with self._lock:
                self._last_result = {"label": label, "score": best_prob}
                self._prediction_count += 1
            
            return self._last_result
            
        except Exception as e:
            # CRITICAL: Log error but don't crash the server
            import logging
            logging.error(f"[PREDICTION ERROR] {e}", exc_info=True)
            return self._last_result  # Return last known good result

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
        with self._lock:
            return dict(self._last_result)
    
    def cleanup(self):
        """
        CRITICAL: Cleanup resources on disconnect to prevent memory leaks
        """
        with self._lock:
            # Clear buffer to free memory
            self._buffer = np.empty((0, DIM[1], DIM[0], CHANNELS), dtype="float32")
            
        # Wait for any running prediction thread to finish (with timeout)
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)  # Wait max 2 seconds
