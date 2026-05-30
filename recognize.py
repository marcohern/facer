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


def main():
    database.init_db()
    known = pipeline.KnownFaces.load()
    print(f"Loaded {len(known)} encoding(s) from the database.")
    if len(known) == 0:
        print("No faces enrolled yet. Run enroll.py first; everyone will show as Unknown.")

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera index {config.CAMERA_INDEX}.")
        return 1

    frame_count = 0
    last_matches = []  # list of pipeline.Match, reused between detection frames

    print("Recognizer running. Press Q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # Keep the loop alive even if a frame read fails — the camera
                # should stay "always on".
                continue

            frame_count += 1
            if frame_count % config.DETECT_EVERY_N_FRAMES == 0:
                last_matches = known.identify(frame)

            for m in last_matches:
                pipeline.draw_label(frame, m.bbox, m.label, m.color)

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
