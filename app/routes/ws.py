from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse
import asyncio
import json
import logging
import time

from app.streaming.stream_manager import StreamManager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/stream/{camera_id}")
async def stream_endpoint(websocket: WebSocket, camera_id: str):
    manager: StreamManager = websocket.app.state.stream_manager
    worker = manager.get_worker(camera_id)

    if not worker:
        logger.warning(f"WebSocket connect to {camera_id}: worker not found")
        await websocket.accept()
        await websocket.send_text(json.dumps({"error": "camera_not_found"}))
        await websocket.close()
        return

    await websocket.accept()
    logger.info(f"WebSocket client connected to {camera_id}")

    queue: asyncio.Queue = asyncio.Queue(maxsize=2)
    worker.add_subscriber(queue)
    frames_sent = 0

    try:
        frame_bytes, metadata, frame_id = worker.get_latest_frame()
        if frame_bytes:
            await websocket.send_bytes(frame_bytes)
            await websocket.send_text(json.dumps(metadata))
            frames_sent += 1

        while True:
            try:
                new_frame_id = await asyncio.wait_for(queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"heartbeat": True}))
                if not worker.is_online:
                    await websocket.send_text(json.dumps({"status": "offline"}))
                continue

            frame_bytes, metadata, _ = worker.get_latest_frame()
            if frame_bytes:
                await websocket.send_bytes(frame_bytes)
                await websocket.send_text(json.dumps(metadata))
                frames_sent += 1

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected from {camera_id} (sent {frames_sent} frames)")
    except Exception as e:
        logger.error(f"WebSocket error for {camera_id} after {frames_sent} frames: {e}")
    finally:
        worker.remove_subscriber(queue)


@router.get("/stream/mjpeg/{camera_id}")
async def mjpeg_stream(request: Request, camera_id: str):
    """MJPEG HTTP stream endpoint - works in any browser via <img> tag."""
    manager: StreamManager = request.app.state.stream_manager
    worker = manager.get_worker(camera_id)

    if not worker:
        return {"error": "camera_not_found"}

    async def frame_generator():
        last_frame_id = -1
        while True:
            frame_bytes, metadata, frame_id = worker.get_latest_frame()
            if frame_bytes and frame_id != last_frame_id:
                last_frame_id = frame_id
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )
            await asyncio.sleep(0.04)  # ~25fps max

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.websocket("/ws/status")
async def status_endpoint(websocket: WebSocket):
    await websocket.accept()
    manager: StreamManager = websocket.app.state.stream_manager

    try:
        while True:
            status = manager.get_all_status()
            summary = {}
            for cam_id, info in status.items():
                summary[cam_id] = {
                    "is_online": info["is_online"],
                    "fps": info["stats"].get("current_fps", 0),
                }
            await websocket.send_text(json.dumps({"cameras": summary}))
            await asyncio.sleep(2.0)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Status WebSocket error: {e}")
