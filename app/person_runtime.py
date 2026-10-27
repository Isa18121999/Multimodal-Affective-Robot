"""Runtime integration for temporary multi-person tracking.

No biometric templates or face images are persisted. A tracker ID exists only
while a face remains visible and may be associated with the name spoken during
the active interaction.
"""

from __future__ import annotations

import os
from functools import wraps

import cv2

from src.identity.person_tracker import PersonTracker


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
            # The largest visible face is the deterministic interaction candidate.
            selected = max(tracks, key=lambda t: t.bbox[2] * t.bbox[3])
            if robot.active_person_id not in {t.person_id for t in tracks}:
                robot.active_person_id = selected.person_id
        else:
            robot.active_person_id = None
        return bool(tracks)

    robot._detect_face = detect_face_with_people

    original_interact = robot._interact

    @wraps(original_interact)
    async def interact_with_person():
        # Preserve the person selected immediately before the interaction.
        selected_id = robot.active_person_id
        robot._active_person_for_interaction = selected_id
        await original_interact()
        robot._active_person_for_interaction = None

    robot._interact = interact_with_person

    original_listen = robot._listen

    @wraps(original_listen)
    async def listen_with_name_association():
        text = await original_listen()
        selected_id = getattr(robot, "_active_person_for_interaction", None)
        # The first listening turn is the name question. Associate only when a
        # valid name is extracted; later conversational turns are ignored.
        if selected_id and not getattr(robot, "_person_name_associated", False):
            extractor = getattr(robot, "_extract_name", None)
            name = extractor(text) if extractor else None
            if name and tracker.associate_name(selected_id, name):
                robot.current_child_name = name
                robot.current_people = tracker.snapshot()
                robot._person_name_associated = True
        return text

    robot._listen = listen_with_name_association
    robot._person_name_associated = False

    original_start = robot.start

    @wraps(original_start)
    async def start_with_person_reset():
        tracker.clear_names()
        tracker.tracks.clear()
        robot.current_people = []
        robot.active_person_id = None
        robot._person_name_associated = False
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
        await original_stop()

    robot.stop = stop_with_person_reset
