import numpy as np

from src.identity.person_tracker import PersonTracker


class FakeCascade:
    def __init__(self, faces):
        self.faces = faces

    def empty(self):
        return False

    def detectMultiScale(self, gray, **kwargs):
        return np.asarray(self.faces, dtype=np.int32).reshape((-1, 4))


def make_tracker(faces, **kwargs):
    tracker = object.__new__(PersonTracker)
    tracker.cascade = FakeCascade(faces)
    tracker.max_missing_seconds = kwargs.get("max_missing_seconds", 2.0)
    tracker.max_match_distance = kwargs.get("max_match_distance", 140.0)
    tracker.tracks = {}
    tracker._next_number = 1
    return tracker


def test_assigns_stable_ids_to_nearby_faces():
    tracker = make_tracker([(20, 20, 100, 100), (220, 20, 100, 100)])
    frame = np.zeros((300, 400, 3), dtype=np.uint8)

    first = tracker.update(frame)
    assert [p.person_id for p in first] == ["person_1", "person_2"]

    tracker.cascade.faces = [(28, 22, 100, 100), (212, 22, 100, 100)]
    second = tracker.update(frame)
    assert {p.person_id for p in second} == {"person_1", "person_2"}


def test_one_track_is_not_assigned_to_two_nearby_detections():
    tracker = make_tracker([(20, 20, 100, 100)])
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    first = tracker.update(frame)
    first_id = first[0].person_id

    tracker.cascade.faces = [(24, 20, 100, 100), (45, 20, 100, 100)]
    second = tracker.update(frame)

    assert len({track.person_id for track in second}) == 2
    assert first_id in {track.person_id for track in second}


def test_disappeared_person_gets_new_id_after_track_expires(monkeypatch):
    tracker = make_tracker([(20, 20, 100, 100)], max_missing_seconds=0.5)
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    first = tracker.update(frame)[0]
    assert tracker.associate_name(first.person_id, "Mateo")

    ticks = iter([100.0, 101.0, 102.0])
    monkeypatch.setattr("src.identity.person_tracker.time.monotonic", lambda: next(ticks))
    tracker.cascade.faces = []
    assert tracker.update(frame) == []
    # The previous temporary track has expired, so the same location gets a new ID.
    tracker.cascade.faces = [(20, 20, 100, 100)]
    returned = tracker.update(frame)
    assert returned[0].person_id != first.person_id
    assert returned[0].name is None


def test_name_is_session_scoped_and_validated():
    tracker = make_tracker([])
    track = tracker._new_track((20, 20, 100, 100), 0.0)

    assert tracker.associate_name(track.person_id, "  Mateo   Pérez ")
    assert tracker.tracks[track.person_id].name == "Mateo Pérez"
    assert not tracker.associate_name("missing", "Ana")
    assert not tracker.associate_name(track.person_id, "   ")

    tracker.clear_names()
    assert tracker.tracks[track.person_id].name is None


def test_clear_removes_tracks_and_names():
    tracker = make_tracker([])
    track = tracker._new_track((20, 20, 100, 100), 0.0)
    tracker.associate_name(track.person_id, "Ana")

    tracker.clear()

    assert tracker.snapshot() == []
