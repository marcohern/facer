"""Facer desktop app — enroll users and test recognition from the camera.

A single Tkinter window with two tabs (Enroll / Recognize) that share one camera
feed. All face logic is delegated to the shared `pipeline` module, so this GUI
behaves identically to the `enroll.py` / `recognize.py` command-line tools.

Run:
    python app.py
"""

import sys

from FacerSplash import FacerSplash
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

import config
import database
import face_engine
import pipeline
from VideoCaptureService import VideoCaptureService

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

        # Enrollment session state.
        self._current_user_id = None
        self._captured = 0

        # Active tab name, mirrored as a plain string so the detect callbacks
        # never have to call into (non-thread-safe) Tkinter.
        self._active_mode = "Enroll"

        # Keep references to PhotoImages so Tk doesn't garbage-collect them.
        self._enroll_photo = None
        self._recognize_photo = None

        # The service owns the camera + its detection thread; we start in Enroll
        # mode (draw face boxes) and swap the callback on tab change.
        self.service = VideoCaptureService(
            config.CAMERA_INDEX, detect_fn=self._enroll_detect
        )
        if not self.service.is_opened():
            messagebox.showerror(
                "Camera error",
                f"Could not open camera index {config.CAMERA_INDEX}.\n"
                "Check that a webcam is connected and not in use by another app.",
            )

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        if self.service.is_opened():
            self.service.start()
            self.root.after(FRAME_INTERVAL_MS, self._update_frame)

    # ----- UI construction -------------------------------------------------

    def _build_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        self._build_enroll_tab()
        self._build_recognize_tab()

        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        if not self.service.is_opened():
            # No camera: leave the tabs visible but inert.
            self.enroll_status.config(text="Camera unavailable.")
            self.recognize_status.config(text="Camera unavailable.")

    def _build_enroll_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Enroll")

        self.enroll_video = ttk.Label(tab)
        self.enroll_video.pack(padx=8, pady=8)

        form = ttk.Frame(tab)
        form.pack(fill="x", padx=8)
        ttk.Label(form, text="Name:").grid(row=0, column=0, sticky="e", padx=4, pady=2)
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(form, textvariable=self.name_var, width=30)
        self.name_entry.grid(row=0, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(form, text="Email:").grid(row=1, column=0, sticky="e", padx=4, pady=2)
        self.email_var = tk.StringVar()
        self.email_entry = ttk.Entry(form, textvariable=self.email_var, width=30)
        self.email_entry.grid(row=1, column=1, sticky="w", padx=4, pady=2)

        buttons = ttk.Frame(tab)
        buttons.pack(fill="x", padx=8, pady=6)
        self.capture_btn = ttk.Button(buttons, text="Capture", command=self._on_capture)
        self.capture_btn.pack(side="left")
        ttk.Button(buttons, text="New person / Reset", command=self._on_reset).pack(
            side="left", padx=6
        )

        self.enroll_status = ttk.Label(tab, text="Enter a name, then Capture.")
        self.enroll_status.pack(fill="x", padx=8, pady=(0, 8))

    def _build_recognize_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Recognize")

        self.recognize_video = ttk.Label(tab)
        self.recognize_video.pack(padx=8, pady=8)

        ttk.Button(
            tab, text="Capture result", command=self._on_capture_result
        ).pack(padx=8, pady=4)

        self.result_text = tk.Text(tab, height=6, width=50, state="disabled")
        self.result_text.pack(fill="x", padx=8, pady=4)

        self.recognize_status = ttk.Label(tab, text="Live recognition running.")
        self.recognize_status.pack(fill="x", padx=8, pady=(0, 8))

    # ----- camera + detection ----------------------------------------------

    def _active_tab(self):
        return self.notebook.tab(self.notebook.select(), "text")

    def _update_frame(self):
        """UI-thread loop: grab a frame, overlay results, show in active tab."""
        if self._closing or not self.service.is_opened():
            return

        frame = self.service.latest_frame()
        if frame is not None:
            result = self.service.latest_result()

            if self._active_mode == "Enroll":
                for bbox in (result or []):
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), pipeline.COLOR_KNOWN, 2)
                self._enroll_photo = self._to_photo(frame)
                self.enroll_video.config(image=self._enroll_photo)
            else:
                for m in (result or []):
                    pipeline.draw_label(frame, m.bbox, m.label, m.color)
                self._recognize_photo = self._to_photo(frame)
                self.recognize_video.config(image=self._recognize_photo)

        self.root.after(FRAME_INTERVAL_MS, self._update_frame)

    # Detection callbacks handed to the service; each returns the result the
    # matching branch of _update_frame knows how to draw.
    def _enroll_detect(self, frame):
        return [f.bbox for f in face_engine.detect(frame)]

    def _recognize_detect(self, frame):
        return self.known.identify(frame)

    @staticmethod
    def _to_photo(frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    def _grab_frame(self):
        return self.service.latest_frame()

    # ----- enroll actions ---------------------------------------------------

    def _on_capture(self):
        frame = self._grab_frame()
        if frame is None:
            self.enroll_status.config(text="No camera frame yet.")
            return

        if self._current_user_id is None:
            name = self.name_var.get().strip()
            if not name:
                self.enroll_status.config(text="Enter a name before capturing.")
                return
            email = self.email_var.get().strip() or None
            self._current_user_id = database.add_user(name, email)
            self._captured = 0
            self.name_entry.config(state="disabled")
            self.email_entry.config(state="disabled")

        ok, message = pipeline.enroll_frame(self._current_user_id, frame)
        if ok:
            self._captured += 1
            self.enroll_status.config(
                text=f"Enrolling '{self.name_var.get().strip()}' — captured {self._captured}."
            )
        else:
            self.enroll_status.config(text=message)

    def _on_reset(self):
        self._current_user_id = None
        self._captured = 0
        self.name_entry.config(state="normal")
        self.email_entry.config(state="normal")
        self.name_var.set("")
        self.email_var.set("")
        self.enroll_status.config(text="Enter a name, then Capture.")

    # ----- recognize actions ------------------------------------------------

    def _on_tab_changed(self, _event):
        self._active_mode = self._active_tab()
        # Reload the index when entering Recognize so faces just enrolled count,
        # then swap the service's detection to match the active tab.
        if self._active_mode == "Recognize":
            self.known.reload()
            self.service.set_detect_fn(self._recognize_detect)
        else:
            self.service.set_detect_fn(self._enroll_detect)

    def _on_capture_result(self):
        frame = self._grab_frame()
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
