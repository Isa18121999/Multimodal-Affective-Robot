import numpy as np

from src.identity.person_tracker import PersonTracker


class FakeCascade:
    def __init__(self, faces):
        self.faces = faces

    def empty(self):
        return False

    def detectMultiScale(self, gray, **kwargs):
        return np.asarray(self.faces, dtype=np.int32)


def test_assigns_stable_ids_to_nearby_faces(monkeypatch):
    tracker = object.__new__(PersonTracker)
    tracker.cascade = FakeCascade([(20, 20, 100, 100), (220, 20, 100, 100)])
    tracker.max_missing_seconds = 2.0
    tracker.max_match_distance = 140.0
    tracker.tracks = {}
    tracker._next_number = 1

    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    first = tracker.update(frame)
    assert [p.person_id for p in first] == ["person_1", "person_2"]

    tracker.cascade.faces = [(28, 22, 100, 100), (212, 22, 100, 100)]
    second = tracker.update(frame)
    assert {p.person_id for p in second} == {"person_1", "person_2"}


def test_name_is_session_scoped():
    tracker = object.__new__(PersonTracker)
    tracker.tracks = {}
    tracker._next_number = 1
    track = tracker._new_track((20, 20, 100, 100), 0.0)

    assert tracker.associate_name(track.person_id, "  Mateo ") is True
    assert tracker.tracks[track.person_id].name == "Mateo"

    tracker.clear_names()
    assert tracker.tracks[track.person_id].name is None
