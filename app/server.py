# # app/server.py
# # -*- coding: utf-8 -*-
# # Python 3.7

# import socketio
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware

# from .config import DIM
# from .labels import LABELS
# from .utils import b64_to_bgr_image, preprocess_frame_bgr_exact
# from .model_runtime import load_runtime_model, VideoInferenceService


# # ------------------------------------------------------------------------------
# # Base FastAPI app (kept for /healthz, /labels routes)
# # ------------------------------------------------------------------------------
# fastapi_app = FastAPI(title="ASL Realtime Backend (Socket.IO)", version="1.0.0")

# fastapi_app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
# )

# _MODEL = None


# @fastapi_app.on_event("startup")
# def _startup():
#     global _MODEL
#     _MODEL = load_runtime_model()


# @fastapi_app.get("/healthz")
# def health():
#     return {"status": "ok", "model": _MODEL.__class__.__name__}


# @fastapi_app.get("/labels")
# def get_labels():
#     return LABELS


# # ------------------------------------------------------------------------------
# # Socket.IO Server
# # ------------------------------------------------------------------------------
# sio = socketio.AsyncServer(
#     async_mode="asgi",
#     cors_allowed_origins="*",
# )

# # The final exposed ASGI app combines FastAPI + Socket.IO
# app = socketio.ASGIApp(sio, fastapi_app)

# # Track one VideoInferenceService per connected client
# clients = {}


# # ------------------------------------------------------------------------------
# # Socket.IO Events
# # ------------------------------------------------------------------------------
# @sio.event
# async def connect(sid, environ):
#     # Create inference session for client
#     clients[sid] = VideoInferenceService(_MODEL)
#     await sio.emit("status", {"message": "ready"}, to=sid)


# @sio.event
# async def disconnect(sid):
#     # Remove services on disconnect
#     if sid in clients:
#         del clients[sid]


# @sio.event
# async def frame(sid, data):
#     """
#     Client sends:
#        sio.emit("frame", { image: "<base64JPEG>" })

#     Server responds:
#        sio.emit("prediction", { label: "...", score: ... })
#     """

#     svc = clients.get(sid)
#     if svc is None:
#         return

#     try:
#         img_bgr = b64_to_bgr_image(data["image"])
#         frame_norm = preprocess_frame_bgr_exact(img_bgr, DIM)
#         svc.add_frame_already_normalized(frame_norm)
#     except Exception:
#         # We intentionally avoid sending error spam for real-time streaming
#         return

#     # Trigger prediction exactly like your original code (only if previous thread done)
#     svc.maybe_launch_prediction()

#     # Return *latest* prediction every frame (same semantics as showing gloss_show on screen)
#     result = svc.current_result()

#     await sio.emit("prediction", {
#         "label": result["label"],
#         "score": result["score"]
#     }, to=sid)

# app/server.py
# -*- coding: utf-8 -*-
# Python 3.7
import os
import logging
import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    DIM, LOG_LEVEL, CORS_ORIGINS, 
    SOCKETIO_PING_TIMEOUT, SOCKETIO_PING_INTERVAL,
    MAX_CLIENTS
)
from .labels import LABELS
from .utils import b64_to_bgr_image, preprocess_frame_bgr_exact
from .model_runtime import load_runtime_model, VideoInferenceService


# ------------------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.getLevelName(LOG_LEVEL.upper()),
    format="[%(asctime)s] [PID:%(process)d] [%(levelname)s] %(message)s"
)
log = logging.getLogger("asl-socketio")

# ------------------------------------------------------------------------------
# FastAPI Base App (REST)
# ------------------------------------------------------------------------------
fastapi_app = FastAPI(title="ASL Realtime Backend (Socket.IO)", version="1.1.0")

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_MODEL = None


@fastapi_app.on_event("startup")
def _startup():
    global _MODEL
    _MODEL = load_runtime_model()
    log.info(f"Model loaded: {_MODEL.__class__.__name__}")
    log.info(f"Allowed CORS origins: {CORS_ORIGINS}")


@fastapi_app.get("/healthz")
def health():
    log.debug("Health check endpoint hit")
    return {"status": "ok", "model": _MODEL.__class__.__name__}


@fastapi_app.get("/labels")
def get_labels():
    log.debug("Labels requested")
    return LABELS


# ------------------------------------------------------------------------------
# Socket.IO Server Configuration
# ------------------------------------------------------------------------------



# CRITICAL FIX: Proper Socket.IO configuration to prevent crashes
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else "*",
    ping_timeout=SOCKETIO_PING_TIMEOUT,
    ping_interval=SOCKETIO_PING_INTERVAL,
    max_http_buffer_size=10 * 1024 * 1024,  # 10MB max for base64 images
    # Reduce logging noise
    engineio_logger=False,
    logger=False,
)

# Combine FastAPI and Socket.IO
app = socketio.ASGIApp(sio, fastapi_app)
# app = socketio.ASGIApp(
#     sio,
#     fastapi_app,
#     socketio_path="socket.io"
# )
# Track one inference service per connected client
clients = {}


# ------------------------------------------------------------------------------
# Socket.IO Events
# ------------------------------------------------------------------------------

@sio.event
async def connect(sid, environ):
    # CRITICAL FIX: Rate limiting to prevent server overload
    if len(clients) >= MAX_CLIENTS:
        log.warning(f"[CONNECT] Max clients ({MAX_CLIENTS}) reached, rejecting {sid}")
        await sio.emit("error", {"message": "Server at capacity, please try again later"}, to=sid)
        await sio.disconnect(sid)
        return
    
    clients[sid] = VideoInferenceService(_MODEL)
    log.info(f"[CONNECT] Client {sid} connected from {environ.get('REMOTE_ADDR')} (Total: {len(clients)})")
    await sio.emit("status", {"message": "ready"}, to=sid)


@sio.event
async def disconnect(sid):
    if sid in clients:
        # CRITICAL FIX: Cleanup resources to prevent memory leaks
        try:
            clients[sid].cleanup()
        except Exception as e:
            log.error(f"[CLEANUP ERROR] Failed to cleanup client {sid}: {e}")
        
        del clients[sid]
        log.info(f"[DISCONNECT] Client {sid} disconnected and cleaned up")
    else:
        log.warning(f"[DISCONNECT] Unknown SID: {sid}")


@sio.event
async def frame(sid, data):
    """
    Client Event: 'frame'
    ---------------------
    Payload:
        { "image": "<base64 JPEG>" }

    Server emits:
        'prediction' → { "label": "<word>", "score": <float> }
    """
    svc = clients.get(sid)
    if svc is None:
        log.warning(f"[WARN] Received frame from unknown client {sid}")
        return

    try:
        img_bgr = b64_to_bgr_image(data["image"])
        frame_norm = preprocess_frame_bgr_exact(img_bgr, DIM)
        svc.add_frame_already_normalized(frame_norm)
        # REDUCED LOGGING: Only log at debug level to prevent log spam
        # log.debug(f"[FRAME] Received frame from {sid}, buffer size: {svc._buffer.shape[0]}")
    except Exception as e:
        log.error(f"[ERROR] Frame decode/preprocess failed for {sid}: {e}")
        return  # Skip frame silently to avoid client flood

    # Launch prediction thread (only when idle)
    launched = svc.maybe_launch_prediction()

    # Get current latest result
    result = svc.current_result()
    label, score = result["label"], result["score"]

    # REDUCED LOGGING: Only log when prediction actually changes or is significant
    if launched or (label != "none" and score > 0.6):
        log.info(f"[PREDICTION] Client {sid}: {label} ({score:.2f})")

    await sio.emit("prediction", {"label": label, "score": score}, to=sid)
