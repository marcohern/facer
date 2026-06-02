"""Facer desktop app — enroll users and test recognition from the camera.

A single Tkinter window with two tabs (Enroll / Recognize) that share one camera
feed. All face logic is delegated to the shared `VideoCapturePipeline` layer, so this GUI
behaves identically to the `Enroll.py` / `Recognize.py` command-line tools.

Run:
    python app.py
"""

import sys

import tkinter as tk

from facer.FacerSplash import FacerSplash
from facer.FacerApp import FacerApp


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
