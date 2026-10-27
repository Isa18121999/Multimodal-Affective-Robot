"""Privacy-preserving, temporary multi-person face tracking.

This module assigns short-lived IDs based on face-box position over time. It is
not biometric identification: after a track disappears, a returning person may
receive a new ID. No images or face templates are persisted.
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
        track = PersonTrack(f"person_{self._next_number}", bbox, now)
        self._next_number += 1
        self.tracks[track.person_id] = track
        return track

    def update(self, frame: np.ndarray) -> list[PersonTrack]:
        """Detect faces, match boxes one-to-one, and expire stale temporary IDs."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
        )
        detections = [tuple(map(int, face)) for face in faces]
        now = time.monotonic()

        # Greedy global nearest-pair matching prevents one track being assigned
        # to two detections in the same frame.
        candidates = []
        for person_id, track in self.tracks.items():
            for index, bbox in enumerate(detections):
                x, y, w, h = bbox
                center = (x + w / 2, y + h / 2)
                distance = self._distance(center, track.center)
                if distance <= self.max_match_distance:
                    candidates.append((distance, person_id, index))
        candidates.sort(key=lambda item: item[0])

        matched_tracks: set[str] = set()
        matched_detections: set[int] = set()
        assignments: dict[int, PersonTrack] = {}
        for _, person_id, index in candidates:
            if person_id in matched_tracks or index in matched_detections:
                continue
            track = self.tracks[person_id]
            track.bbox = detections[index]
            track.last_seen = now
            matched_tracks.add(person_id)
            matched_detections.add(index)
            assignments[index] = track

        for index, bbox in enumerate(detections):
            if index not in assignments:
                assignments[index] = self._new_track(bbox, now)

        stale = [
            person_id
            for person_id, track in self.tracks.items()
            if now - track.last_seen > self.max_missing_seconds
        ]
        for person_id in stale:
            del self.tracks[person_id]

        # Return only currently visible faces, in the detector's order.
        return [assignments[i] for i in range(len(detections))]

    def associate_name(self, person_id: str, name: str) -> bool:
        track = self.tracks.get(person_id)
        clean_name = " ".join(name.strip().split())
        if not track or not clean_name or len(clean_name) > 30:
            return False
        track.name = clean_name[:1].upper() + clean_name[1:]
        return True

    def clear(self) -> None:
        self.tracks.clear()

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
