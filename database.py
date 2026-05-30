import aiosqlite
import time
import uuid
from typing import Optional
from config import DB_PATH


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS cameras (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                stream_url TEXT NOT NULL UNIQUE,
                is_active INTEGER NOT NULL DEFAULT 1,
                is_online INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS detection_stats (
                camera_id TEXT PRIMARY KEY,
                total_frames INTEGER DEFAULT 0,
                total_faces INTEGER DEFAULT 0,
                total_objects INTEGER DEFAULT 0,
                current_fps REAL DEFAULT 0.0,
                last_frame_at REAL,
                consecutive_failures INTEGER DEFAULT 0,
                FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE
            )
        """)
        await db.commit()


async def create_camera(name: str, stream_url: str) -> dict:
    camera_id = str(uuid.uuid4())
    now = time.time()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO cameras (id, name, stream_url, is_active, is_online, created_at, updated_at) VALUES (?, ?, ?, 1, 0, ?, ?)",
            (camera_id, name, stream_url, now, now)
        )
        await db.execute(
            "INSERT INTO detection_stats (camera_id) VALUES (?)",
            (camera_id,)
        )
        await db.commit()
    return await get_camera(camera_id)


async def get_all_cameras() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("""
            SELECT c.*, s.total_frames, s.total_faces, s.total_objects, 
                   s.current_fps, s.last_frame_at, s.consecutive_failures
            FROM cameras c
            LEFT JOIN detection_stats s ON c.id = s.camera_id
            ORDER BY c.created_at DESC
        """) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def get_camera(camera_id: str) -> Optional[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("""
            SELECT c.*, s.total_frames, s.total_faces, s.total_objects,
                   s.current_fps, s.last_frame_at, s.consecutive_failures
            FROM cameras c
            LEFT JOIN detection_stats s ON c.id = s.camera_id
            WHERE c.id = ?
        """, (camera_id,)) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


async def update_camera(camera_id: str, name: Optional[str] = None, stream_url: Optional[str] = None) -> Optional[dict]:
    now = time.time()
    async with aiosqlite.connect(DB_PATH) as db:
        if name is not None and stream_url is not None:
            await db.execute(
                "UPDATE cameras SET name = ?, stream_url = ?, updated_at = ? WHERE id = ?",
                (name, stream_url, now, camera_id)
            )
        elif name is not None:
            await db.execute(
                "UPDATE cameras SET name = ?, updated_at = ? WHERE id = ?",
                (name, now, camera_id)
            )
        elif stream_url is not None:
            await db.execute(
                "UPDATE cameras SET stream_url = ?, updated_at = ? WHERE id = ?",
                (stream_url, now, camera_id)
            )
        await db.commit()
    return await get_camera(camera_id)


async def delete_camera(camera_id: str) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))
        await db.commit()
        return cursor.rowcount > 0


async def set_camera_active(camera_id: str, is_active: bool):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE cameras SET is_active = ?, updated_at = ? WHERE id = ?",
            (1 if is_active else 0, time.time(), camera_id)
        )
        await db.commit()


async def set_camera_online(camera_id: str, is_online: bool):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE cameras SET is_online = ? WHERE id = ?",
            (1 if is_online else 0, camera_id)
        )
        await db.commit()


async def update_stats(camera_id: str, stats: dict):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            UPDATE detection_stats SET
                total_frames = ?,
                total_faces = ?,
                total_objects = ?,
                current_fps = ?,
                last_frame_at = ?,
                consecutive_failures = ?
            WHERE camera_id = ?
        """, (
            stats.get("total_frames", 0),
            stats.get("total_faces", 0),
            stats.get("total_objects", 0),
            stats.get("current_fps", 0.0),
            stats.get("last_frame_at"),
            stats.get("consecutive_failures", 0),
            camera_id
        ))
        await db.commit()


async def get_active_cameras() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("""
            SELECT c.*, s.total_frames, s.total_faces, s.total_objects,
                   s.current_fps, s.last_frame_at, s.consecutive_failures
            FROM cameras c
            LEFT JOIN detection_stats s ON c.id = s.camera_id
            WHERE c.is_active = 1
            ORDER BY c.created_at DESC
        """) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
