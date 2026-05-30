from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from app.routes.cameras import router as cameras_router
from app.routes.ws import router as ws_router


def create_app() -> FastAPI:
    app = FastAPI(title="OpenCV Camera Server", version="1.0.0")

    app.include_router(cameras_router, prefix="/api")
    app.include_router(ws_router)

    static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(static_dir, "index.html"))

    return app
