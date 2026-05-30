"""Enroll a user's face into the database.

Examples:
    # Capture from the webcam (SPACE to capture, Q to finish):
    python enroll.py --name "Jane Doe" --email jane@example.com

    # Enroll from an existing photo instead of the camera:
    python enroll.py --name "Jane Doe" --email jane@example.com --image jane.jpg
"""

import argparse
import sys

import cv2

import config
import database
import face_engine
import pipeline


def enroll_from_image(user_id, path):
    frame = cv2.imread(path)
    if frame is None:
        print(f"Could not read image: {path}")
        return 0

    ok, message = pipeline.enroll_frame(user_id, frame)
    print(message if ok else f"{message} The image must contain exactly one face.")
    return 1 if ok else 0


def enroll_from_camera(user_id):
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera index {config.CAMERA_INDEX}.")
        return 0

    print("Camera ready. Press SPACE to capture a face, Q to finish.")
    captured = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            faces = face_engine.detect(frame)
            display = frame.copy()
            for f in faces:
                x1, y1, x2, y2 = [int(v) for v in f.bbox]
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            status = f"Faces: {len(faces)} | captured: {captured} | SPACE=capture Q=quit"
            cv2.putText(
                display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2,
            )
            cv2.imshow("Enroll", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                ok, message = pipeline.enroll_frame(user_id, frame)
                if ok:
                    captured += 1
                    print(f"Captured encoding #{captured}.")
                else:
                    print(message)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return captured


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
    user_id = database.add_user(args.name, args.email)
    print(f"User '{args.name}' has id {user_id}.")

    if args.image:
        enroll_from_image(user_id, args.image)
    else:
        enroll_from_camera(user_id)

    total = database.count_encodings(user_id)
    print(f"User '{args.name}' now has {total} encoding(s) stored.")


if __name__ == "__main__":
    sys.exit(main())
