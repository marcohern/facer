# Facer — AI Face Recognition

A Python 3 desktop app that keeps the camera open, continuously scans for faces,
and when it recognizes one, retrieves that user's info from a SQLite database and
overlays it on the live video. Unrecognized faces are labelled **Unknown**.

- **AI engine:** [InsightFace](https://github.com/deepinsight/insightface) `buffalo_l`
  (deep-learning face embeddings via ONNX Runtime).
- **Display:** OpenCV window with bounding boxes + name/email overlay.
- **Storage:** SQLite (`faces.db`).

## Project layout

| File | Purpose |
|------|---------|
| `config.py` | Settings: DB path, camera index, recognition threshold, etc. |
| `database.py` | SQLite layer — `users` and `face_encodings` tables. |
| `FaceEngine.py` | InsightFace wrapper (`FaceEngine` class): detect faces, embed, cosine-match. |
| `pipeline.py` | Shared enroll/identify logic reused by the CLI and the GUI. |
| `Enroll.py` | CLI (`Enroll` class) to capture a face and link it to a user. |
| `Recognize.py` | The always-on recognition app (CLI, `Recognize` class). |
| `app.py` | Desktop GUI: enroll and test recognition in one window. |

> Note: `users.csv` in this folder is sample data and is **not** used by the app.

## Setup

Requires Python 3.8+ and a working webcam.

```bash
pip install -r requirements.txt
```

On the **first run**, InsightFace automatically downloads the `buffalo_l` model
pack (~300 MB) to `~/.insightface`. This happens once.

Using a CUDA GPU? Swap `onnxruntime` for `onnxruntime-gpu` in `requirements.txt`.

## 1. Enroll faces

Capture from the webcam (press **SPACE** to capture, ideally a few times from
slightly different angles; **Q** to finish):

```bash
python Enroll.py --name "Your Name" --email you@example.com
```

Or enroll from an existing photo (must contain exactly one face):

```bash
python Enroll.py --name "Your Name" --email you@example.com --image you.jpg
```

## 2. Run recognition

```bash
python Recognize.py
```

A window opens showing the live camera. Enrolled people get a green box with
their name/email and match score; everyone else shows a red **Unknown** box.
Press **Q** to quit.

## Desktop app (GUI)

Prefer a window over the command line? Run the desktop app, which does both
enrollment and recognition in one place:

```bash
python app.py
```

- **Enroll tab:** type a name (and optional email), then click **Capture** a few
  times from slightly different angles. **New person / Reset** clears the form to
  enroll someone else.
- **Recognize tab:** shows the live camera with labelled boxes. Click **Capture
  result** to test the current frame and list the matched user (name/email/score)
  or **Unknown**. Faces enrolled in this session are picked up automatically when
  you switch to this tab.

## Tuning

Edit `config.py`:

- `RECOGNITION_THRESHOLD` (default `0.5`) — raise it to be stricter (fewer false
  matches), lower it to be more lenient.
- `CAMERA_INDEX` — change if you have multiple cameras.
- `DETECT_EVERY_N_FRAMES` — increase to lighten CPU load on slower machines.
