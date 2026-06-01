"""Enroll tab for the Facer notebook.

Owns its own widgets (video preview + name/email form + capture/reset buttons +
status line) and the enrollment session state. Exposes the trio the notebook
drives every frame/tab-change: `detect()` (what the camera service runs in the
background), `render()` (draw the latest frame + result into this tab's preview),
and `on_enter()` (make the service run *this* tab's detection).

Composition over subclassing (mirrors FacerSplash): the tab content lives on
`self.frame`, which the notebook adds via `notebook.add(tab.frame, ...)`.
"""

import tkinter as tk
from tkinter import ttk

import cv2

import pipeline
from Database import db
from FaceEngine import engine


class VideoNotebookEnrollTab:
    def __init__(self, parent_notebook, service, to_photo):
        self.frame = ttk.Frame(parent_notebook)
        self.service = service
        self._to_photo = to_photo

        # Enrollment session state.
        self._current_user_id = None
        self._captured = 0
        self._photo = None  # keep a ref so Tk doesn't garbage-collect it

        self._build()

    # ----- UI construction -------------------------------------------------

    def _build(self):
        self.video = ttk.Label(self.frame)
        self.video.pack(padx=8, pady=8)

        form = ttk.Frame(self.frame)
        form.pack(fill="x", padx=8)
        ttk.Label(form, text="Name:").grid(row=0, column=0, sticky="e", padx=4, pady=2)
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(form, textvariable=self.name_var, width=30)
        self.name_entry.grid(row=0, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(form, text="Email:").grid(row=1, column=0, sticky="e", padx=4, pady=2)
        self.email_var = tk.StringVar()
        self.email_entry = ttk.Entry(form, textvariable=self.email_var, width=30)
        self.email_entry.grid(row=1, column=1, sticky="w", padx=4, pady=2)

        buttons = ttk.Frame(self.frame)
        buttons.pack(fill="x", padx=8, pady=6)
        self.capture_btn = ttk.Button(buttons, text="Capture", command=self._on_capture)
        self.capture_btn.pack(side="left")
        ttk.Button(buttons, text="New person / Reset", command=self._on_reset).pack(
            side="left", padx=6
        )

        self.status = ttk.Label(self.frame, text="Enter a name, then Capture.")
        self.status.pack(fill="x", padx=8, pady=(0, 8))

    def set_status(self, text):
        self.status.config(text=text)

    # ----- camera + detection ----------------------------------------------

    def detect(self, frame):
        """Background detection: face boxes for the live preview."""
        return [f.bbox for f in engine.detect(frame)]

    def render(self, frame, result):
        """Draw the latest face boxes onto `frame` and show it in the preview."""
        for bbox in (result or []):
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), pipeline.COLOR_KNOWN, 2)
        self._photo = self._to_photo(frame)
        self.video.config(image=self._photo)

    def on_enter(self):
        self.service.set_detect_fn(self.detect)

    # ----- enroll actions ---------------------------------------------------

    def _on_capture(self):
        frame = self.service.latest_frame()
        if frame is None:
            self.status.config(text="No camera frame yet.")
            return

        if self._current_user_id is None:
            name = self.name_var.get().strip()
            if not name:
                self.status.config(text="Enter a name before capturing.")
                return
            email = self.email_var.get().strip() or None
            self._current_user_id = db.add_user(name, email)
            self._captured = 0
            self.name_entry.config(state="disabled")
            self.email_entry.config(state="disabled")

        ok, message = pipeline.enroll_frame(self._current_user_id, frame)
        if ok:
            self._captured += 1
            self.status.config(
                text=f"Enrolling '{self.name_var.get().strip()}' — captured {self._captured}."
            )
        else:
            self.status.config(text=message)

    def _on_reset(self):
        self._current_user_id = None
        self._captured = 0
        self.name_entry.config(state="normal")
        self.email_entry.config(state="normal")
        self.name_var.set("")
        self.email_var.set("")
        self.status.config(text="Enter a name, then Capture.")
