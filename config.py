import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.environ.get("DB_PATH", os.path.join(BASE_DIR, "cameras.db"))

DETECTION_CONF_THRESHOLD = float(os.environ.get("DETECTION_CONF_THRESHOLD", "0.5"))
DETECTION_NMS_THRESHOLD = float(os.environ.get("DETECTION_NMS_THRESHOLD", "0.4"))
SSD_INPUT_SIZE = (300, 300)
SSD_MEAN = (104.0, 177.0, 123.0)

JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "70"))
MAX_FRAME_WIDTH = int(os.environ.get("MAX_FRAME_WIDTH", "640"))

RECONNECT_BASE_DELAY = float(os.environ.get("RECONNECT_BASE_DELAY", "2.0"))
RECONNECT_MAX_DELAY = float(os.environ.get("RECONNECT_MAX_DELAY", "30.0"))

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(BASE_DIR, "models"))

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

STATS_FLUSH_INTERVAL = int(os.environ.get("STATS_FLUSH_INTERVAL", "60"))
