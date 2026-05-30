from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
import asyncio
import json
import logging

from app.streaming.stream_manager import StreamManager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/stream/{camera_id}")
async def stream_endpoint(websocket: WebSocket, camera_id: str):
    manager: StreamManager = websocket.app.state.stream_manager
    worker = manager.get_worker(camera_id)

    if not worker:
        await websocket.accept()
        await websocket.send_text(json.dumps({"error": "camera_not_found"}))
        await websocket.close()
        return

    await websocket.accept()

    queue: asyncio.Queue = asyncio.Queue(maxsize=2)
    worker.add_subscriber(queue)

    try:
        frame_bytes, metadata, frame_id = worker.get_latest_frame()
        if frame_bytes:
            await websocket.send_bytes(frame_bytes)
            await websocket.send_text(json.dumps(metadata))

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

    except WebSocketDisconnect:
        logger.debug(f"WebSocket client disconnected from {camera_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {camera_id}: {e}")
    finally:
        worker.remove_subscriber(queue)


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
