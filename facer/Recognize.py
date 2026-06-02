"""Always-on face recognition app.

Opens the camera, continuously scans for faces, and on a match looks the user
up in the SQLite database and overlays their info. Unknown faces are labelled
"Unknown". Press Q to quit.
"""

import sys

import cv2

import config
from .Database import db
from .VideoCapturePipeline import pipeline, KnownFaces
from .VideoCaptureService import VideoCaptureService


class Recognize:
    """An always-on recognition session over the enrolled face index."""

    def __init__(self):
        self.known = KnownFaces.load()

    def run(self):
        print(f"Loaded {len(self.known)} encoding(s) from the database.")
        if len(self.known) == 0:
            print("No faces enrolled yet. Run Enroll.py first; everyone will show as Unknown.")

        # The service runs identify() continuously on its own thread, so the
        # display loop below stays smooth without the old DETECT_EVERY_N_FRAMES
        # throttle.
        with VideoCaptureService(config.CAMERA_INDEX, detect_fn=self.known.identify) as svc:
            if not svc.is_opened():
                print(f"Could not open camera index {config.CAMERA_INDEX}.")
                return 1
            svc.start()

            print("Recognizer running. Press Q to quit.")
            try:
                while True:
                    frame = svc.latest_frame()
                    if frame is None:
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                        continue

                    for m in (svc.latest_result() or []):
                        pipeline.draw_label(frame, m.bbox, m.label, m.color)

                    cv2.imshow("Face Recognition", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            finally:
                cv2.destroyAllWindows()

        return 0


def main():
    db.init_db()
    return Recognize().run()


if __name__ == "__main__":
    sys.exit(main())
