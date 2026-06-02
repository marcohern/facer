"""Facer GUI shell: wires the camera service, known-faces index, and notebook.

`FacerApp` is the thin Tkinter shell — it owns the `VideoCaptureService` +
`KnownFaces`, composes a `VideoNotebook`, and runs the `root.after()` frame loop.
The root `app.py` launcher constructs it after the splash.
"""

from tkinter import messagebox

import config
from .Database import db
from .VideoCapturePipeline import KnownFaces
from .VideoCaptureService import VideoCaptureService
from .VideoNotebook import VideoNotebook

# How often the display loop refreshes (milliseconds).
FRAME_INTERVAL_MS = 30


class FacerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facer")

        db.init_db()
        self.known = KnownFaces.load()

        # Set once the window is closing, to stop the _update_frame after-loop.
        self._closing = False

        # The service owns the camera + its detection thread; the notebook points
        # it at the active tab's detection (via on_enter) during construction.
        self.service = VideoCaptureService(config.CAMERA_INDEX)
        self.notebook = VideoNotebook(self.root, self.service, self.known)

        if not self.service.is_opened():
            messagebox.showerror(
                "Camera error",
                f"Could not open camera index {config.CAMERA_INDEX}.\n"
                "Check that a webcam is connected and not in use by another app.",
            )
            self.notebook.show_camera_unavailable()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        if self.service.is_opened():
            self.service.start()
            self.root.after(FRAME_INTERVAL_MS, self._update_frame)

    # ----- frame loop -------------------------------------------------------

    def _update_frame(self):
        """UI-thread loop: hand the latest frame + result to the active tab."""
        if self._closing or not self.service.is_opened():
            return

        frame = self.service.latest_frame()
        if frame is not None:
            self.notebook.active_tab().render(frame, self.service.latest_result())

        self.root.after(FRAME_INTERVAL_MS, self._update_frame)

    # ----- shutdown ---------------------------------------------------------

    def _on_close(self):
        self._closing = True
        self.service.stop()
        self.root.destroy()
