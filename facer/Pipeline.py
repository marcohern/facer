"""Shared enrollment + recognition logic used by both the CLI and the GUI.

This is the reuse layer: the rules for "store one face for a user" and "identify
the faces in a frame" live here so the command-line tools (`Enroll.py`,
`Recognize.py`) and the desktop app (`app.py`) can never drift apart.

Only depends on the framework-agnostic backend (`config`, `Database`,
`FaceEngine`) plus cv2/numpy for drawing — no GUI or CLI concerns.
"""

from dataclasses import dataclass

import cv2

import config
from .Database import db
from .FaceEngine import engine

# BGR colors used for known vs. unknown faces (OpenCV uses BGR ordering).
COLOR_KNOWN = (0, 200, 0)    # green
COLOR_UNKNOWN = (0, 0, 255)  # red


@dataclass
class Match:
    """One recognised (or unrecognised) face in a frame."""

    bbox: object          # np.ndarray [x1, y1, x2, y2]
    user: object          # user dict {id, name, email} or None for Unknown
    score: float
    label: str            # text drawn on the box / shown in results
    color: tuple          # BGR color for the box


class KnownFaces:
    """In-memory index of every stored embedding, with cached user lookups.

    Loaded once, then queried per frame. Call `reload()` after new enrollments
    so they become recognisable without restarting.
    """

    def __init__(self, matrix, user_ids):
        self._matrix = matrix
        self._user_ids = user_ids
        self._user_cache = {}  # user_id -> user dict, to avoid repeated DB hits

    @classmethod
    def load(cls):
        matrix, user_ids = db.load_all_encodings()
        return cls(matrix, user_ids)

    def reload(self):
        """Reload all embeddings from the database (e.g. after enrolling)."""
        self._matrix, self._user_ids = db.load_all_encodings()
        self._user_cache.clear()

    def __len__(self):
        return len(self._user_ids)

    def _resolve_user(self, user_id):
        if user_id not in self._user_cache:
            self._user_cache[user_id] = db.get_user(user_id)
        return self._user_cache[user_id]

    def identify(self, frame, threshold=None):
        """Identify every face in a BGR frame. Returns a list of Match."""
        if threshold is None:
            threshold = config.RECOGNITION_THRESHOLD

        matches = []
        for face in engine.detect(frame):
            user_id, score = engine.cosine_match(
                face.normed_embedding, self._matrix, self._user_ids, threshold,
            )
            if user_id is not None:
                user = self._resolve_user(user_id) or {"name": "?", "email": ""}
                label = f"{user['name']} ({score:.2f})"
                if user.get("email"):
                    label += f" {user['email']}"
                matches.append(Match(face.bbox, user, score, label, COLOR_KNOWN))
            else:
                label = f"Unknown ({score:.2f})"
                matches.append(Match(face.bbox, None, score, label, COLOR_UNKNOWN))
        return matches


class Pipeline:
    """Stateless enrollment + drawing helpers shared by every frontend.

    The methods are static — they hold no state, just the shared rules — but the
    whole app calls them through the `pipeline` singleton below (mirroring
    `FaceEngine`'s `engine` and `Database`'s `db`).
    """

    @staticmethod
    def capture_single_face(frame):
        """Detect faces in a BGR frame and enforce exactly one.

        Returns (face, error_message). On success `error_message` is None; on
        failure `face` is None and `error_message` explains why.
        """
        faces = engine.detect(frame)
        if len(faces) == 0:
            return None, "No face detected."
        if len(faces) > 1:
            return None, f"{len(faces)} faces detected — only one person at a time."
        return faces[0], None

    @staticmethod
    def enroll_frame(user_id, frame):
        """Validate exactly one face in `frame` and store its embedding for `user_id`.

        Returns (ok, message).
        """
        face, error = Pipeline.capture_single_face(frame)
        if error:
            return False, error
        db.add_encoding(user_id, face.normed_embedding)
        return True, "Stored 1 encoding."

    @staticmethod
    def draw_label(frame, bbox, text, color):
        """Draw a colored bounding box with a filled text caption onto a BGR frame."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        top = max(y1, th + 6)
        cv2.rectangle(frame, (x1, top - th - 6), (x1 + tw + 6, top), color, -1)
        cv2.putText(
            frame, text, (x1 + 3, top - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
        )


# Shared singleton, mirroring FaceEngine's `engine` and Database's `db`.
pipeline = Pipeline()
