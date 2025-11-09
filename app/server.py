# # -*- coding: utf-8 -*-
# # Python 3.7
# import json
# from typing import Dict

# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from fastapi.middleware.cors import CORSMiddleware
# from starlette.websockets import WebSocketState

# from .config import DIM
# from .labels import LABELS
# # from .utils import b64_to_bgr_image, preprocess_frame_bgr
# # from .model_runtime import load_runtime_model, VideoInferenceService
# from .schemas import WSFrameIn, WSPredictionOut, WSInfoOut, WSErrorOut

# from .utils import b64_to_bgr_image, preprocess_frame_bgr_exact
# from .model_runtime import load_runtime_model, VideoInferenceService


# app = FastAPI(title="ASL Realtime Backend", version="1.0.0")

# # Optional: relax CORS for quick integration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
# )

# _MODEL = None  # global shared model object


# @app.on_event("startup")
# def _startup():
#     global _MODEL
#     _MODEL = load_runtime_model()


# @app.get("/healthz")
# def health() -> Dict[str, str]:
#     return {"status": "ok", "model": _MODEL.__class__.__name__}


# @app.get("/labels")
# def get_labels() -> Dict[int, str]:
#     return LABELS


# @app.websocket("/ws")
# async def ws_endpoint(ws: WebSocket):
#     await ws.accept()
#     svc = VideoInferenceService(_MODEL)

#     # Send hello
#     await ws.send_json(WSInfoOut(message="ready").dict())

#     try:
#         while True:
#             raw = await ws.receive_text()
#             try:
#                 payload = WSFrameIn.parse_raw(raw)
#             except Exception as e:
#                 await ws.send_json(WSErrorOut(message="invalid payload: {}".format(e)).dict())
#                 continue

#             try:
#                 img_bgr = b64_to_bgr_image(payload.data)
#                 frame_norm = preprocess_frame_bgr_exact(img_bgr, DIM)  # resize + /255.0; BGR preserved
#                 svc.add_frame_already_normalized(frame_norm)
#             except Exception as e:
#                 await ws.send_json(WSErrorOut(message="decode/preprocess error: {}".format(e)).dict())
#                 continue
#             # Only when buffer is full, try launching a prediction thread (exact semantics)
#             svc.maybe_launch_prediction()

#             # Non-blocking: always return the latest result; client will see updates as they arrive
#             res = svc.current_result()
#             await ws.send_json({
#                 "type": "prediction",
#                 "label": res["label"],
#                 "score": res["score"]
#             })
#             # if svc.ready():
#             #     result = await svc.predict()
#             #     await ws.send_json({
#             #         "type": "prediction",
#             #         "label": result["label"],
#             #         "score": result["score"]
#             #     })


#     except WebSocketDisconnect:
#         # graceful close
#         if ws.application_state != WebSocketState.DISCONNECTED:
#             await ws.close()
#     except Exception as e:
#         # hard error
#         try:
#             await ws.send_json(WSErrorOut(message="server error: {}".format(e)).dict())
#         finally:
#             if ws.application_state != WebSocketState.DISCONNECTED:
#                 await ws.close()
# app/server.py
# -*- coding: utf-8 -*-
# Python 3.7

import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import DIM
from .labels import LABELS
from .utils import b64_to_bgr_image, preprocess_frame_bgr_exact
from .model_runtime import load_runtime_model, VideoInferenceService


# ------------------------------------------------------------------------------
# Base FastAPI app (kept for /healthz, /labels routes)
# ------------------------------------------------------------------------------
fastapi_app = FastAPI(title="ASL Realtime Backend (Socket.IO)", version="1.0.0")

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

_MODEL = None


@fastapi_app.on_event("startup")
def _startup():
    global _MODEL
    _MODEL = load_runtime_model()


@fastapi_app.get("/healthz")
def health():
    return {"status": "ok", "model": _MODEL.__class__.__name__}


@fastapi_app.get("/labels")
def get_labels():
    return LABELS


# ------------------------------------------------------------------------------
# Socket.IO Server
# ------------------------------------------------------------------------------
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
)

# The final exposed ASGI app combines FastAPI + Socket.IO
app = socketio.ASGIApp(sio, fastapi_app)

# Track one VideoInferenceService per connected client
clients = {}


# ------------------------------------------------------------------------------
# Socket.IO Events
# ------------------------------------------------------------------------------
@sio.event
async def connect(sid, environ):
    # Create inference session for client
    clients[sid] = VideoInferenceService(_MODEL)
    await sio.emit("status", {"message": "ready"}, to=sid)


@sio.event
async def disconnect(sid):
    # Remove services on disconnect
    if sid in clients:
        del clients[sid]


@sio.event
async def frame(sid, data):
    """
    Client sends:
       sio.emit("frame", { image: "<base64JPEG>" })

    Server responds:
       sio.emit("prediction", { label: "...", score: ... })
    """

    svc = clients.get(sid)
    if svc is None:
        return

    try:
        img_bgr = b64_to_bgr_image(data["image"])
        frame_norm = preprocess_frame_bgr_exact(img_bgr, DIM)
        svc.add_frame_already_normalized(frame_norm)
    except Exception:
        # We intentionally avoid sending error spam for real-time streaming
        return

    # Trigger prediction exactly like your original code (only if previous thread done)
    svc.maybe_launch_prediction()

    # Return *latest* prediction every frame (same semantics as showing gloss_show on screen)
    result = svc.current_result()

    await sio.emit("prediction", {
        "label": result["label"],
        "score": result["score"]
    }, to=sid)
