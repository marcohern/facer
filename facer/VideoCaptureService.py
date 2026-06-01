"""Threaded camera capture with its own background detection loop.

`VideoCaptureService` owns a single `cv2.VideoCapture` and runs two daemon
threads: a *reader* that keeps the most recent BGR frame, and a *detector* that
runs an injected `detect_fn(frame)` on that frame and keeps the most recent
result. Both are exposed through thread-safe accessors so a frontend only has to
render frames and results — it never touches the camera or threading directly.

The detection callback is swappable at runtime (`set_detect_fn`), which lets a
single service back several modes (e.g. the GUI's Enroll vs. Recognize tabs)
without the service knowing anything about what detection means.

Depends only on cv2 / threading / config — never on `FaceEngine` or
`Pipeline`, so it stays decoupled from any particular detection logic.
"""

import threading

import cv2

import config


class VideoCaptureService:
    def __init__(self, camera_index=config.CAMERA_INDEX, detect_fn=None):
        self._cap = cv2.VideoCapture(camera_index)
        self._detect_fn = detect_fn

        self._lock = threading.Lock()
        self._frame = None          # most recent BGR frame from the camera
        self._result = None         # most recent detect_fn(frame) output
        self._epoch = 0             # bumped on set_detect_fn to drop stale results
        self._stop = threading.Event()

        self._reader = None
        self._detector = None

    def is_opened(self):
        return self._cap is not None and self._cap.isOpened()

    def start(self):
        """Spawn the reader + detector threads. Returns False if the camera
        could not be opened (no threads are started)."""
        if not self.is_opened():
            return False
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._detector = threading.Thread(target=self._detect_loop, daemon=True)
        self._reader.start()
        self._detector.start()
        return True

    def set_detect_fn(self, fn):
        """Swap the detection callback. Drops any pending result so a renderer
        never sees a result produced by the previous callback."""
        with self._lock:
            self._detect_fn = fn
            self._result = None
            self._epoch += 1

    def latest_frame(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def latest_result(self):
        with self._lock:
            return self._result

    def stop(self):
        self._stop.set()
        for t in (self._reader, self._detector):
            if t is not None:
                t.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False

    # ----- worker threads --------------------------------------------------

    def _read_loop(self):
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
            else:
                # Keep the camera "always on" even if a read momentarily fails.
                self._stop.wait(0.01)

    def _detect_loop(self):
        while not self._stop.is_set():
            with self._lock:
                fn = self._detect_fn
                epoch = self._epoch
                frame = None if self._frame is None else self._frame.copy()

            if fn is None or frame is None:
                self._stop.wait(0.05)
                continue

            result = fn(frame)
            with self._lock:
                # Only publish if the callback hasn't been swapped meanwhile,
                # so a result of the old kind can't land after a mode switch.
                if epoch == self._epoch:
                    self._result = result
