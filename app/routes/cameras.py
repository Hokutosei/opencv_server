from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
import time

import database
from models import CameraCreate, CameraUpdate, CameraResponse, DetectionStats
from app.streaming.stream_manager import StreamManager

router = APIRouter()


def _format_camera_response(cam: dict, request: Request) -> dict:
    stats = None
    if cam.get("total_frames") is not None:
        stats = DetectionStats(
            total_frames=cam.get("total_frames", 0),
            total_faces=cam.get("total_faces", 0),
            total_objects=cam.get("total_objects", 0),
            current_fps=cam.get("current_fps", 0.0),
            last_frame_at=cam.get("last_frame_at"),
            consecutive_failures=cam.get("consecutive_failures", 0),
        )

    base_url = str(request.base_url).rstrip("/")
    ws_scheme = "wss" if request.url.scheme == "https" else "ws"
    ws_url = f"{ws_scheme}://{request.headers.get('host', 'localhost')}/ws/stream/{cam['id']}"

    return CameraResponse(
        id=cam["id"],
        name=cam["name"],
        stream_url=cam["stream_url"],
        is_active=bool(cam["is_active"]),
        is_online=bool(cam["is_online"]),
        created_at=cam["created_at"],
        updated_at=cam["updated_at"],
        stats=stats,
        stream_ws_url=ws_url,
    )


def _get_manager(request: Request) -> StreamManager:
    return request.app.state.stream_manager


@router.get("/cameras")
async def list_cameras(request: Request):
    cameras = await database.get_all_cameras()
    manager = _get_manager(request)
    live_status = manager.get_all_status()

    results = []
    for cam in cameras:
        cam_id = cam["id"]
        if cam_id in live_status:
            cam["is_online"] = live_status[cam_id]["is_online"]
            live_stats = live_status[cam_id]["stats"]
            cam["current_fps"] = live_stats.get("current_fps", cam.get("current_fps", 0))
            cam["last_frame_at"] = live_stats.get("last_frame_at", cam.get("last_frame_at"))
        results.append(_format_camera_response(cam, request))
    return results


@router.post("/cameras", status_code=201)
async def register_camera(data: CameraCreate, request: Request):
    try:
        cam = await database.create_camera(data.name, data.stream_url)
    except Exception as e:
        if "UNIQUE" in str(e):
            raise HTTPException(status_code=409, detail="A camera with this stream URL already exists")
        raise HTTPException(status_code=500, detail=str(e))

    manager = _get_manager(request)
    await manager.start_camera(cam["id"], cam["stream_url"])
    await database.set_camera_active(cam["id"], True)

    return _format_camera_response(cam, request)


@router.get("/cameras/{camera_id}")
async def get_camera(camera_id: str, request: Request):
    cam = await database.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    manager = _get_manager(request)
    worker = manager.get_worker(camera_id)
    if worker:
        live_stats = worker.stats
        cam["is_online"] = worker.is_online
        cam["current_fps"] = live_stats.get("current_fps", 0)
        cam["last_frame_at"] = live_stats.get("last_frame_at")
        cam["total_frames"] = live_stats.get("total_frames", cam.get("total_frames", 0))
        cam["total_faces"] = live_stats.get("total_faces", cam.get("total_faces", 0))
        cam["total_objects"] = live_stats.get("total_objects", cam.get("total_objects", 0))

    return _format_camera_response(cam, request)


@router.put("/cameras/{camera_id}")
async def update_camera(camera_id: str, data: CameraUpdate, request: Request):
    cam = await database.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    manager = _get_manager(request)
    url_changed = data.stream_url is not None and data.stream_url != cam["stream_url"]

    if url_changed:
        await manager.stop_camera(camera_id)

    updated = await database.update_camera(camera_id, name=data.name, stream_url=data.stream_url)

    if url_changed and updated["is_active"]:
        await manager.start_camera(camera_id, updated["stream_url"])

    return _format_camera_response(updated, request)


@router.delete("/cameras/{camera_id}")
async def delete_camera(camera_id: str, request: Request):
    manager = _get_manager(request)
    await manager.stop_camera(camera_id)

    deleted = await database.delete_camera(camera_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Camera not found")
    return {"detail": "Camera deleted"}


@router.post("/cameras/{camera_id}/start")
async def start_camera(camera_id: str, request: Request):
    cam = await database.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    manager = _get_manager(request)
    await manager.start_camera(camera_id, cam["stream_url"])
    await database.set_camera_active(camera_id, True)
    return {"detail": "Camera started"}


@router.post("/cameras/{camera_id}/stop")
async def stop_camera(camera_id: str, request: Request):
    cam = await database.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    manager = _get_manager(request)
    worker = manager.get_worker(camera_id)
    if worker:
        await database.update_stats(camera_id, worker.stats)
    await manager.stop_camera(camera_id)
    await database.set_camera_active(camera_id, False)
    return {"detail": "Camera stopped"}


@router.get("/health")
async def health(request: Request):
    manager = _get_manager(request)
    status = manager.get_all_status()
    return {
        "status": "ok",
        "active_cameras": len(status),
        "online_cameras": sum(1 for s in status.values() if s["is_online"]),
    }
