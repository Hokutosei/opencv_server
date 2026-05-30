import cv2
import os
import ssl
import urllib.request
import logging

from config import MODELS_DIR, SSD_INPUT_SIZE, SSD_MEAN, DETECTION_CONF_THRESHOLD, DETECTION_NMS_THRESHOLD

logger = logging.getLogger(__name__)

COCO_LABELS = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"


def _download_file(url: str, dest: str):
    """Download a file with SSL workaround for Windows cert issues."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "Python/OpenCV-Server"})
    with urllib.request.urlopen(req, context=ctx) as response:
        with open(dest, "wb") as f:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)


def ensure_models():
    """Ensure SSD model files exist. Returns (prototxt_path, model_path) or (None, None) if download fails."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    prototxt_path = os.path.join(MODELS_DIR, "deploy.prototxt")
    model_path = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

    if not os.path.exists(prototxt_path):
        logger.info("Downloading deploy.prototxt...")
        try:
            _download_file(PROTOTXT_URL, prototxt_path)
            logger.info("Downloaded deploy.prototxt")
        except Exception as e:
            logger.warning(f"Failed to download deploy.prototxt: {e}")
            logger.warning("Object detection disabled. Only Haar cascade face detection available.")
            if os.path.exists(prototxt_path):
                os.remove(prototxt_path)
            return None, None

    if not os.path.exists(model_path):
        logger.info("Downloading SSD caffemodel (~10MB)...")
        try:
            _download_file(MODEL_URL, model_path)
            logger.info("Downloaded SSD caffemodel")
        except Exception as e:
            logger.warning(f"Failed to download SSD model: {e}")
            logger.warning("Object detection disabled. Only Haar cascade face detection available.")
            if os.path.exists(model_path):
                os.remove(model_path)
            return prototxt_path, None

    return prototxt_path, model_path


class ObjectDetector:
    def __init__(self, prototxt_path: str = None, model_path: str = None):
        if prototxt_path is None or model_path is None:
            prototxt_path, model_path = ensure_models()

        if model_path is None or prototxt_path is None:
            self._model = None
            logger.warning("ObjectDetector initialized in disabled mode (no model)")
            return

        self._model = cv2.dnn.DetectionModel(prototxt_path, model_path)
        self._model.setInputSize(SSD_INPUT_SIZE)
        self._model.setInputMean(SSD_MEAN)
        self._model.setInputSwapRB(False)
        self._model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def detect(self, frame, conf_threshold: float = None) -> list[dict]:
        """
        Detect objects in a BGR frame.
        Returns list of dicts: {class_id, label, confidence, bbox: (x, y, w, h)}
        """
        if self._model is None:
            return []

        if conf_threshold is None:
            conf_threshold = DETECTION_CONF_THRESHOLD

        class_ids, confidences, boxes = self._model.detect(
            frame,
            confThreshold=conf_threshold,
            nmsThreshold=DETECTION_NMS_THRESHOLD
        )

        results = []
        if class_ids is not None and len(class_ids) > 0:
            for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), boxes):
                label = COCO_LABELS[class_id] if class_id < len(COCO_LABELS) else f"class_{class_id}"
                results.append({
                    "class_id": int(class_id),
                    "label": label,
                    "confidence": float(confidence),
                    "bbox": (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                })
        return results
