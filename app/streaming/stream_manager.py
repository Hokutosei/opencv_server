import logging
import asyncio
from typing import Optional

from app.detection.face_detector import FaceDetector
from app.streaming.camera_worker import CameraWorker
import database

logger = logging.getLogger(__name__)


class StreamManager:
    def __init__(self):
        self._workers: dict[str, CameraWorker] = {}
        self._face_detector = FaceDetector()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def _on_status_change(self, camera_id: str, online: bool):
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                database.set_camera_online(camera_id, online),
                self._loop
            )

    async def start_camera(self, camera_id: str, stream_url: str):
        if camera_id in self._workers:
            logger.warning(f"Camera {camera_id} already running")
            return

        worker = CameraWorker(camera_id, stream_url, self._face_detector)
        worker.set_status_callback(self._on_status_change)
        worker.start()
        self._workers[camera_id] = worker
        logger.info(f"Started camera worker: {camera_id}")

    async def stop_camera(self, camera_id: str):
        worker = self._workers.pop(camera_id, None)
        if worker:
            worker.stop()
            logger.info(f"Stopped camera worker: {camera_id}")

    def get_worker(self, camera_id: str) -> Optional[CameraWorker]:
        return self._workers.get(camera_id)

    def get_all_status(self) -> dict:
        status = {}
        for cam_id, worker in self._workers.items():
            status[cam_id] = {
                "is_online": worker.is_online,
                "stats": worker.stats
            }
        return status

    async def start_all_active(self):
        cameras = await database.get_active_cameras()
        for cam in cameras:
            await self.start_camera(cam["id"], cam["stream_url"])
        logger.info(f"Started {len(cameras)} active cameras")

    async def shutdown(self):
        logger.info("Shutting down all camera workers...")
        camera_ids = list(self._workers.keys())
        for cam_id in camera_ids:
            await self.stop_camera(cam_id)

        for cam_id in camera_ids:
            worker_stats = None
            worker = self._workers.get(cam_id)
            if worker:
                worker_stats = worker.stats
            if worker_stats:
                try:
                    await database.update_stats(cam_id, worker_stats)
                except Exception as e:
                    logger.error(f"Failed to save stats for {cam_id}: {e}")
        logger.info("All camera workers stopped")
