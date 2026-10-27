"""Runtime integration for temporary multi-person tracking.

No biometric templates or face images are persisted. A tracker ID exists only
while a face remains visible and may be associated with the name spoken during
the active interaction.
"""

from __future__ import annotations

import asyncio
import os
from functools import wraps

import cv2

from src.identity.person_tracker import PersonTracker
from app.websocket_manager import ws_manager


def install_person_tracking(robot) -> None:
    """Attach multi-person tracking to the existing RobotController instance."""
    if getattr(robot, "_person_tracking_installed", False):
        return

    cascade_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "haarcascade_frontalface_default.xml",
    )
    if not os.path.exists(cascade_path):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    tracker = PersonTracker(cascade_path)
    robot.person_tracker = tracker
    robot.current_people = []
    robot.active_person_id = None
    robot._person_tracking_installed = True
    robot._person_event_loop = None

    original_detect_face = robot._detect_face

    def detect_face_with_people():
        robot._init_camera()
        if robot.cap is None:
            robot.current_people = []
            return False
        ret, frame = robot.cap.read()
        if not ret:
            robot.current_people = []
            return False

        tracks = tracker.update(frame)
        robot.current_people = tracker.snapshot()
        if tracks:
            selected = max(tracks, key=lambda t: t.bbox[2] * t.bbox[3])
            if robot.active_person_id not in {t.person_id for t in tracks}:
                robot.active_person_id = selected.person_id
        else:
            robot.active_person_id = None

        loop = getattr(robot, "_person_event_loop", None)
        if loop and not loop.is_closed():
            payload = {
                "people": robot.current_people,
                "active_person_id": robot.active_person_id,
            }
            asyncio.run_coroutine_threadsafe(
                ws_manager.broadcast("robot_people", payload), loop
            )
        return bool(tracks)

    robot._detect_face = detect_face_with_people

    original_interact = robot._interact

    @wraps(original_interact)
    async def interact_with_person():
        selected_id = robot.active_person_id
        robot._active_person_for_interaction = selected_id
        robot._person_name_associated = False

        track = next((t for t in tracker.tracks.values() if t.person_id == selected_id), None)
        known_name = track.name if track else None
        original_greeting = robot.GREETING
        original_extract = robot._extract_name

        if known_name:
            robot.GREETING = f"¡Hola, {known_name}! Qué alegría verte otra vez. ¿Cómo estás?"

            def extract_known_name(_text, _known_name=known_name):
                return _known_name

            robot._extract_name = extract_known_name

        try:
            await original_interact()
        finally:
            robot.GREETING = original_greeting
            robot._extract_name = original_extract
            robot._active_person_for_interaction = None
            robot._person_name_associated = False

    robot._interact = interact_with_person

    original_listen = robot._listen

    @wraps(original_listen)
    async def listen_with_name_association():
        text = await original_listen()
        selected_id = getattr(robot, "_active_person_for_interaction", None)
        if selected_id and not getattr(robot, "_person_name_associated", False):
            extractor = getattr(robot, "_extract_name", None)
            name = extractor(text) if extractor else None
            if name and tracker.associate_name(selected_id, name):
                robot.current_child_name = name
                robot.current_people = tracker.snapshot()
                robot._person_name_associated = True
                loop = getattr(robot, "_person_event_loop", None)
                if loop and not loop.is_closed():
                    asyncio.run_coroutine_threadsafe(
                        ws_manager.broadcast(
                            "robot_people",
                            {
                                "people": robot.current_people,
                                "active_person_id": robot.active_person_id,
                                "active_person_name": name,
                            },
                        ),
                        loop,
                    )
        return text

    robot._listen = listen_with_name_association

    original_start = robot.start

    @wraps(original_start)
    async def start_with_person_reset():
        tracker.clear_names()
        tracker.tracks.clear()
        robot.current_people = []
        robot.active_person_id = None
        robot._person_name_associated = False
        robot._person_event_loop = asyncio.get_running_loop()
        await original_start()

    robot.start = start_with_person_reset

    original_stop = robot.stop

    @wraps(original_stop)
    async def stop_with_person_reset():
        tracker.clear_names()
        tracker.tracks.clear()
        robot.current_people = []
        robot.active_person_id = None
        robot._person_name_associated = False
        robot._person_event_loop = None
        await original_stop()

    robot.stop = stop_with_person_reset
