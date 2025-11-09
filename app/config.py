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
