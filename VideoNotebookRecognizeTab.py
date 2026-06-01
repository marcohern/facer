"""Recognize tab for the Facer notebook.

Owns its own widgets (live video preview + "Capture result" button + result text
+ status line). Exposes the same trio the notebook drives: `detect()` (run by the
camera service in the background — here, `KnownFaces.identify`), `render()` (draw
labelled matches into the preview), and `on_enter()` (reload the index so faces
just enrolled count, then point the service at this tab's detection).

Composition over subclassing (mirrors FacerSplash): the tab content lives on
`self.frame`, which the notebook adds via `notebook.add(tab.frame, ...)`.
"""

import tkinter as tk
from tkinter import ttk

from Pipeline import pipeline


class VideoNotebookRecognizeTab:
    def __init__(self, parent_notebook, service, known, to_photo):
        self.frame = ttk.Frame(parent_notebook)
        self.service = service
        self.known = known
        self._to_photo = to_photo

        self._photo = None  # keep a ref so Tk doesn't garbage-collect it

        self._build()

    # ----- UI construction -------------------------------------------------

    def _build(self):
        self.video = ttk.Label(self.frame)
        self.video.pack(padx=8, pady=8)

        ttk.Button(
            self.frame, text="Capture result", command=self._on_capture_result
        ).pack(padx=8, pady=4)

        self.result_text = tk.Text(self.frame, height=6, width=50, state="disabled")
        self.result_text.pack(fill="x", padx=8, pady=4)

        self.status = ttk.Label(self.frame, text="Live recognition running.")
        self.status.pack(fill="x", padx=8, pady=(0, 8))

    def set_status(self, text):
        self.status.config(text=text)

    # ----- camera + detection ----------------------------------------------

    def detect(self, frame):
        """Background detection: identify every face in the frame."""
        return self.known.identify(frame)

    def render(self, frame, result):
        """Draw labelled matches onto `frame` and show it in the preview."""
        for m in (result or []):
            pipeline.draw_label(frame, m.bbox, m.label, m.color)
        self._photo = self._to_photo(frame)
        self.video.config(image=self._photo)

    def on_enter(self):
        # Reload the index when entering so faces just enrolled count, then point
        # the service at this tab's detection.
        self.known.reload()
        self.service.set_detect_fn(self.detect)

    # ----- recognize actions ------------------------------------------------

    def _on_capture_result(self):
        frame = self.service.latest_frame()
        if frame is None:
            self._set_result("No camera frame yet.")
            return

        matches = self.known.identify(frame)
        if not matches:
            self._set_result("No face detected.")
            return

        lines = []
        for m in matches:
            if m.user is not None:
                email = m.user.get("email") or ""
                lines.append(f"MATCH: {m.user['name']}  {email}  (score {m.score:.2f})")
            else:
                lines.append(f"UNKNOWN  (best score {m.score:.2f})")
        self._set_result("\n".join(lines))

    def _set_result(self, text):
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text)
        self.result_text.config(state="disabled")
