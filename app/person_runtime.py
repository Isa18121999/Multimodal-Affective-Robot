"""Direct runtime integration for temporary face tracks and spoken names.

Tracking IDs/names exist only in memory while a track is visible. This is not
biometric identity recognition, and no photographs or face templates are saved.
"""
from __future__ import annotations

from pathlib import Path

import cv2

from src.identity.person_tracker import PersonTracker


def create_person_tracker() -> PersonTracker:
    root = Path(__file__).resolve().parent.parent
    cascade = root / "haarcascade_frontalface_default.xml"
    if not cascade.exists():
        cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    return PersonTracker(str(cascade))


def publish_people(robot) -> None:
    """Schedule a dashboard update from the controller's running event loop."""
    loop = getattr(robot, "_person_event_loop", None)
    if not loop or loop.is_closed():
        return

    import asyncio
    from app.websocket_manager import ws_manager

    payload = {
        "people": robot.current_people,
        "active_person_id": robot.active_person_id,
        "active_person_name": robot.current_child_name,
    }
    try:
        asyncio.run_coroutine_threadsafe(
            ws_manager.broadcast("robot_people", payload), loop
        )
    except RuntimeError:
        # The event loop may have stopped between checking and scheduling.
        return
