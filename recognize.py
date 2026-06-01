"""Always-on face recognition app.

Opens the camera, continuously scans for faces, and on a match looks the user
up in the SQLite database and overlays their info. Unknown faces are labelled
"Unknown". Press Q to quit.
"""

import sys

import cv2

import config
import database
import pipeline
from VideoCaptureService import VideoCaptureService


def main():
    database.init_db()
    known = pipeline.KnownFaces.load()
    print(f"Loaded {len(known)} encoding(s) from the database.")
    if len(known) == 0:
        print("No faces enrolled yet. Run enroll.py first; everyone will show as Unknown.")

    # The service runs identify() continuously on its own thread, so the display
    # loop below stays smooth without the old DETECT_EVERY_N_FRAMES throttle.
    with VideoCaptureService(config.CAMERA_INDEX, detect_fn=known.identify) as svc:
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


if __name__ == "__main__":
    sys.exit(main())
