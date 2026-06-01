"""Facer desktop app — enroll users and test recognition from the camera.

A single Tkinter window with two tabs (Enroll / Recognize) that share one camera
feed. All face logic is delegated to the shared `pipeline` module, so this GUI
behaves identically to the `enroll.py` / `recognize.py` command-line tools.

Run:
    python app.py
"""

import sys

from FacerSplash import FacerSplash
import tkinter as tk
from tkinter import messagebox

import config
import database
import pipeline
from VideoCaptureService import VideoCaptureService
from VideoNotebook import VideoNotebook

# How often the display loop refreshes (milliseconds).
FRAME_INTERVAL_MS = 30


class FacerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facer")

        database.init_db()
        self.known = pipeline.KnownFaces.load()

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


def _show_splash(root, on_done):
    splash = FacerSplash(root)
    splash.show(on_done)


def main():
    root = tk.Tk()
    root.withdraw()  # hidden until the splash finishes

    def start_app():
        FacerApp(root)
        root.deiconify()

    _show_splash(root, start_app)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
