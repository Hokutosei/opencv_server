import uvicorn
import asyncio
import logging
from contextlib import asynccontextmanager

from config import HOST, PORT, STATS_FLUSH_INTERVAL
import database
from app.api import create_app
from app.streaming.stream_manager import StreamManager
from app.detection.object_detector import ensure_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

_stats_task = None


async def _periodic_stats_flush(manager: StreamManager):
    while True:
        await asyncio.sleep(STATS_FLUSH_INTERVAL)
        for cam_id, worker_info in manager.get_all_status().items():
            try:
                await database.update_stats(cam_id, worker_info["stats"])
            except Exception as e:
                logger.error(f"Failed to flush stats for {cam_id}: {e}")


@asynccontextmanager
async def lifespan(app):
    global _stats_task

    logger.info("Ensuring detection models are available...")
    ensure_models()

    logger.info("Initializing database...")
    await database.init_db()

    manager = StreamManager()
    manager.set_event_loop(asyncio.get_running_loop())
    app.state.stream_manager = manager

    logger.info("Starting active cameras...")
    await manager.start_all_active()

    _stats_task = asyncio.create_task(_periodic_stats_flush(manager))

    logger.info(f"Server ready at http://{HOST}:{PORT}")
    yield

    logger.info("Shutting down...")
    if _stats_task:
        _stats_task.cancel()
        try:
            await _stats_task
        except asyncio.CancelledError:
            pass

    await manager.shutdown()
    logger.info("Shutdown complete")


app = create_app()
app.router.lifespan_context = lifespan


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        log_level="info",
        reload=False,
    )
