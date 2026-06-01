"""Tabbed notebook for the Facer GUI.

Wraps a `ttk.Notebook` holding the Enroll and Recognize tabs, and routes the
camera service's detection to whichever tab is active: on every tab change it
calls the new tab's `on_enter()` (which swaps the service's `detect_fn`). The
hosting app asks `active_tab()` each frame and lets that tab render itself.

Also home to the shared `to_photo()` helper, injected into both tabs so they can
share it without importing this module back (which would be a circular import).

Composition over subclassing (mirrors FacerSplash): holds `self.notebook`.
"""

from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from VideoNotebookEnrollTab import VideoNotebookEnrollTab
from VideoNotebookRecognizeTab import VideoNotebookRecognizeTab


def to_photo(frame_bgr):
    """Convert a BGR frame to a Tk-displayable PhotoImage."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


class VideoNotebook:
    def __init__(self, parent, service, known):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True)

        self.enroll_tab = VideoNotebookEnrollTab(self.notebook, service, to_photo)
        self.notebook.add(self.enroll_tab.frame, text="Enroll")

        self.recognize_tab = VideoNotebookRecognizeTab(
            self.notebook, service, known, to_photo
        )
        self.notebook.add(self.recognize_tab.frame, text="Recognize")

        # Look up the tab object for whatever frame the notebook reports selected.
        self._tabs = {
            str(self.enroll_tab.frame): self.enroll_tab,
            str(self.recognize_tab.frame): self.recognize_tab,
        }

        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # Start on Enroll; set its detection explicitly rather than relying on the
        # tab-changed event firing during construction.
        self.active = self.enroll_tab
        self.enroll_tab.on_enter()

    def _on_tab_changed(self, _event):
        self.active = self._tabs[self.notebook.select()]
        self.active.on_enter()

    def active_tab(self):
        return self.active

    def show_camera_unavailable(self):
        self.enroll_tab.set_status("Camera unavailable.")
        self.recognize_tab.set_status("Camera unavailable.")
