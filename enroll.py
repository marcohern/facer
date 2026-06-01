"""Enroll a user's face into the database.

Examples:
    # Capture from the webcam (SPACE to capture, Q to finish):
    python Enroll.py --name "Jane Doe" --email jane@example.com

    # Enroll from an existing photo instead of the camera:
    python Enroll.py --name "Jane Doe" --email jane@example.com --image jane.jpg
"""

import argparse
import sys

import cv2

import config
import database
import pipeline
from FaceEngine import engine
from VideoCaptureService import VideoCaptureService


class Enroll:
    """An enrollment session for one person.

    Creating an `Enroll` registers the user in the database; `from_image` /
    `from_camera` then attach face encodings to that user.
    """

    def __init__(self, name, email=None):
        self.name = name
        self.email = email
        self.user_id = database.add_user(name, email)
        self.captured = 0

    def from_image(self, path):
        frame = cv2.imread(path)
        if frame is None:
            print(f"Could not read image: {path}")
            return 0

        ok, message = pipeline.enroll_frame(self.user_id, frame)
        print(message if ok else f"{message} The image must contain exactly one face.")
        if ok:
            self.captured += 1
        return 1 if ok else 0

    def from_camera(self):
        # The detect thread supplies live face boxes for the overlay; SPACE
        # enrolls the latest frame (enroll_frame re-detects on it, so the count
        # is exact).
        with VideoCaptureService(config.CAMERA_INDEX, detect_fn=engine.detect) as svc:
            if not svc.is_opened():
                print(f"Could not open camera index {config.CAMERA_INDEX}.")
                return 0
            svc.start()

            print("Camera ready. Press SPACE to capture a face, Q to finish.")
            try:
                while True:
                    frame = svc.latest_frame()
                    if frame is None:
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                        continue

                    faces = svc.latest_result() or []
                    display = frame.copy()
                    for f in faces:
                        x1, y1, x2, y2 = [int(v) for v in f.bbox]
                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    status = f"Faces: {len(faces)} | captured: {self.captured} | SPACE=capture Q=quit"
                    cv2.putText(
                        display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2,
                    )
                    cv2.imshow("Enroll", display)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord(" "):
                        ok, message = pipeline.enroll_frame(self.user_id, frame)
                        if ok:
                            self.captured += 1
                            print(f"Captured encoding #{self.captured}.")
                        else:
                            print(message)
            finally:
                cv2.destroyAllWindows()

        return self.captured

    def encoding_count(self):
        return database.count_encodings(self.user_id)


def main():
    parser = argparse.ArgumentParser(description="Enroll a face into the database.")
    parser.add_argument("--name", required=True, help="User's display name.")
    parser.add_argument("--email", default=None, help="User's email (optional, unique).")
    parser.add_argument(
        "--image", default=None,
        help="Path to a photo to enroll from instead of the camera.",
    )
    args = parser.parse_args()

    database.init_db()
    enroller = Enroll(args.name, args.email)
    print(f"User '{enroller.name}' has id {enroller.user_id}.")

    if args.image:
        enroller.from_image(args.image)
    else:
        enroller.from_camera()

    print(f"User '{enroller.name}' now has {enroller.encoding_count()} encoding(s) stored.")


if __name__ == "__main__":
    sys.exit(main())
