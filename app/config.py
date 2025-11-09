import os

# Model / preprocessing settings aligned with your script
DIM = (224, 224)
FRAMES = 10
CHANNELS = 3
THRESHOLD = 0.50

# Allow overriding via env vars without code changes
MODEL_PATH = os.getenv("MODEL_PATH", "./model/WLASL20c_model.h5")

# In tests or dev, you can force a dummy model that emits stable predictions
USE_DUMMY_MODEL = bool(int(os.getenv("USE_DUMMY_MODEL", "0")))

# CORS Configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

# CRITICAL: TensorFlow Threading Configuration
# Limit TensorFlow threads to prevent CPU overload and crashes
TF_INTER_OP_THREADS = int(os.getenv("TF_INTER_OP_THREADS", "1"))
TF_INTRA_OP_THREADS = int(os.getenv("TF_INTRA_OP_THREADS", "2"))

# Socket.IO Configuration
SOCKETIO_PING_TIMEOUT = int(os.getenv("SOCKETIO_PING_TIMEOUT", "60"))
SOCKETIO_PING_INTERVAL = int(os.getenv("SOCKETIO_PING_INTERVAL", "25"))

# Maximum concurrent clients (optional rate limiting)
MAX_CLIENTS = int(os.getenv("MAX_CLIENTS", "100"))
