"""Privacy-preserving person tracking for the CHUWI runtime.

This module does not perform biometric identification or store face images/templates.
It assigns temporary IDs to visible faces and can associate a spoken name with the
current temporary ID for the active interaction/session only.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PersonTrack:
    person_id: str
    bbox: tuple[int, int, int, int]
    last_seen: float
    name: str | None = None

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return x + w / 2, y + h / 2


class PersonTracker:
    """Lightweight multi-person tracker based on face boxes and temporal proximity."""

    def __init__(
        self,
        face_cascade_path: str,
        max_missing_seconds: float = 2.0,
        max_match_distance: float = 140.0,
    ):
        self.cascade = cv2.CascadeClassifier(face_cascade_path)
        if self.cascade.empty():
            raise FileNotFoundError(f"No se pudo cargar Haar Cascade: {face_cascade_path}")
        self.max_missing_seconds = max_missing_seconds
        self.max_match_distance = max_match_distance
        self.tracks: dict[str, PersonTrack] = {}
        self._next_number = 1

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _new_track(self, bbox: tuple[int, int, int, int], now: float) -> PersonTrack:
        track = PersonTrack(
            person_id=f"person_{self._next_number}",
            bbox=bbox,
            last_seen=now,
        )
        self._next_number += 1
        self.tracks[track.person_id] = track
        return track

    def update(self, frame: np.ndarray) -> list[PersonTrack]:
        """Detect visible faces and keep stable temporary IDs while they remain visible."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60),
        )
        detections = [tuple(map(int, face)) for face in faces]
        now = time.monotonic()
        unmatched = set(self.tracks)
        current: list[PersonTrack] = []

        for bbox in detections:
            x, y, w, h = bbox
            center = (x + w / 2, y + h / 2)
            best_id = None
            best_distance = float("inf")
            for person_id in unmatched:
                distance = self._distance(center, self.tracks[person_id].center)
                if distance < best_distance:
                    best_distance = distance
                    best_id = person_id

            if best_id is not None and best_distance <= self.max_match_distance:
                track = self.tracks[best_id]
                unmatched.remove(best_id)
                track.bbox = bbox
                track.last_seen = now
            else:
                track = self._new_track(bbox, now)
            current.append(track)

        stale = [
            person_id
            for person_id, track in self.tracks.items()
            if now - track.last_seen > self.max_missing_seconds
        ]
        for person_id in stale:
            del self.tracks[person_id]

        return current

    def associate_name(self, person_id: str, name: str) -> bool:
        """Associate a name with the current temporary tracker ID only."""
        track = self.tracks.get(person_id)
        if not track:
            return False
        track.name = name.strip().capitalize()
        return True

    def clear_names(self) -> None:
        for track in self.tracks.values():
            track.name = None

    def snapshot(self) -> list[dict]:
        return [
            {
                "person_id": track.person_id,
                "name": track.name,
                "bbox": list(track.bbox),
                "recognized": track.name is not None,
            }
            for track in self.tracks.values()
        ]
