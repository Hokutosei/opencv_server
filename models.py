from pydantic import BaseModel, Field, HttpUrl
from typing import Optional
import time


class CameraCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    stream_url: str = Field(..., description="MJPEG stream URL (e.g., http://192.168.1.50:81/stream)")


class CameraUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    stream_url: Optional[str] = None


class DetectionStats(BaseModel):
    total_frames: int = 0
    total_faces: int = 0
    total_objects: int = 0
    current_fps: float = 0.0
    last_frame_at: Optional[float] = None
    consecutive_failures: int = 0


class CameraResponse(BaseModel):
    id: str
    name: str
    stream_url: str
    is_active: bool
    is_online: bool
    created_at: float
    updated_at: float
    stats: Optional[DetectionStats] = None
    stream_ws_url: Optional[str] = None


class DetectionResult(BaseModel):
    type: str
    label: str
    confidence: float
    bbox: list
