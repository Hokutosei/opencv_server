import cv2
import threading
import asyncio
import time
import random
import logging
import numpy as np

from config import (
    MAX_FRAME_WIDTH, JPEG_QUALITY,
    RECONNECT_BASE_DELAY, RECONNECT_MAX_DELAY
)
from app.detection.face_detector import FaceDetector
from app.detection.object_detector import ObjectDetector

logger = logging.getLogger(__name__)


class CameraWorker:
    def __init__(self, camera_id: str, stream_url: str, face_detector: FaceDetector):
        self.camera_id = camera_id
        self.stream_url = stream_url
        self._face_detector = face_detector
        self._object_detector: ObjectDetector = None

        self._running = threading.Event()
        self._thread: threading.Thread = None

        self._frame_lock = threading.Lock()
        self._latest_frame: bytes = None
        self._latest_metadata: dict = None
        self._frame_id: int = 0

        self._subscribers: set[asyncio.Queue] = set()
        self._subscribers_lock = threading.Lock()

        self._is_online = False
        self._stats = {
            "total_frames": 0,
            "total_faces": 0,
            "total_objects": 0,
            "current_fps": 0.0,
            "last_frame_at": None,
            "consecutive_failures": 0,
        }

        self._on_status_change = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"cam-{self.camera_id[:8]}")
        self._thread.start()

    def stop(self):
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._is_online = False

    @property
    def is_online(self) -> bool:
        return self._is_online

    @property
    def stats(self) -> dict:
        return self._stats.copy()

    def get_latest_frame(self) -> tuple[bytes, dict, int]:
        with self._frame_lock:
            return self._latest_frame, self._latest_metadata, self._frame_id

    def add_subscriber(self, queue: asyncio.Queue):
        with self._subscribers_lock:
            self._subscribers.add(queue)

    def remove_subscriber(self, queue: asyncio.Queue):
        with self._subscribers_lock:
            self._subscribers.discard(queue)

    def set_status_callback(self, callback):
        self._on_status_change = callback

    def _notify_online(self, online: bool):
        self._is_online = online
        if self._on_status_change:
            try:
                self._on_status_change(self.camera_id, online)
            except Exception as e:
                logger.error(f"Status callback error: {e}")

    def _run(self):
        logger.info(f"CameraWorker {self.camera_id} started: {self.stream_url}")

        try:
            self._object_detector = ObjectDetector()
        except Exception as e:
            logger.error(f"Failed to load object detector: {e}")
            return

        failures = 0

        while self._running.is_set():
            cap = None
            try:
                cap = cv2.VideoCapture(self.stream_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                if not cap.isOpened():
                    raise RuntimeError("Failed to open stream")

                failures = 0
                self._stats["consecutive_failures"] = 0
                self._notify_online(True)
                logger.info(f"Camera {self.camera_id} connected")

                fps_counter = 0
                fps_timer = time.time()
                current_fps = 0.0

                while self._running.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning(f"Camera {self.camera_id}: read failed")
                        break

                    frame = self._resize_frame(frame)

                    faces = self._face_detector.detect(frame)
                    objects = self._object_detector.detect(frame)

                    self._draw_detections(frame, faces, objects)

                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                    _, jpeg = cv2.imencode(".jpg", frame, encode_params)
                    jpeg_bytes = jpeg.tobytes()

                    fps_counter += 1
                    elapsed = time.time() - fps_timer
                    if elapsed >= 1.0:
                        current_fps = fps_counter / elapsed
                        fps_counter = 0
                        fps_timer = time.time()

                    now = time.time()
                    metadata = {
                        "frame_id": self._frame_id + 1,
                        "timestamp": now,
                        "fps": round(current_fps, 1),
                        "face_count": len(faces),
                        "object_count": len(objects),
                        "detections": (
                            [{"type": "face", "label": "face", "confidence": 1.0, "bbox": list(b)} for b in faces] +
                            [{"type": "object", "label": o["label"], "confidence": round(o["confidence"], 3), "bbox": list(o["bbox"])} for o in objects]
                        )
                    }

                    with self._frame_lock:
                        self._latest_frame = jpeg_bytes
                        self._latest_metadata = metadata
                        self._frame_id += 1

                    self._stats["total_frames"] += 1
                    self._stats["total_faces"] += len(faces)
                    self._stats["total_objects"] += len(objects)
                    self._stats["current_fps"] = current_fps
                    self._stats["last_frame_at"] = now

                    with self._subscribers_lock:
                        for queue in self._subscribers:
                            try:
                                queue.put_nowait(self._frame_id)
                            except asyncio.QueueFull:
                                pass

            except Exception as e:
                logger.warning(f"Camera {self.camera_id} error: {e}")
            finally:
                if cap is not None:
                    cap.release()

            if not self._running.is_set():
                break

            failures += 1
            self._stats["consecutive_failures"] = failures
            self._notify_online(False)

            delay = min(RECONNECT_BASE_DELAY * (2 ** (failures - 1)), RECONNECT_MAX_DELAY)
            jitter = random.uniform(0, delay * 0.5)
            delay += jitter

            logger.info(f"Camera {self.camera_id} reconnecting in {delay:.1f}s (attempt {failures})")
            self._running.wait(delay)

        self._notify_online(False)
        logger.info(f"CameraWorker {self.camera_id} stopped")

    def _resize_frame(self, frame):
        h, w = frame.shape[:2]
        if w > MAX_FRAME_WIDTH:
            scale = MAX_FRAME_WIDTH / w
            new_w = MAX_FRAME_WIDTH
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return frame

    def _draw_detections(self, frame, faces, objects):
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for obj in objects:
            bx, by, bw, bh = obj["bbox"]
            label = obj["label"]
            conf = obj["confidence"]
            color = (255, 0, 0) if label == "person" else (0, 0, 255)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)
            text = f"{label} {conf:.2f}"
            cv2.putText(frame, text, (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
