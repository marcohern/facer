# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -r requirements.txt          # install deps (Python 3.8+, needs a webcam)

python enroll.py --name "Jane Doe" --email jane@example.com          # enroll from webcam (SPACE=capture, Q=finish)
python enroll.py --name "Jane Doe" --email jane@example.com --image jane.jpg   # enroll from a photo (exactly one face)

python recognize.py                       # run the always-on recognizer CLI (Q to quit)

python app.py                             # desktop GUI: enroll + recognize in one window
```

There is no build step, linter, or test suite configured. First run of either script downloads the `buffalo_l` model pack (~300 MB) to `~/.insightface`.

## Architecture

A layered pipeline around InsightFace embeddings stored in SQLite. Data flow: a BGR frame → `face_engine.detect()` → 512-d L2-normalized embedding → cosine match against all stored embeddings → user lookup. Two frontends sit on top — the CLIs (`enroll.py`, `recognize.py`) and the desktop GUI (`app.py`) — and both go through the shared `pipeline.py` layer so their behavior can't diverge.

- **`config.py`** — single source of all tunable settings (DB path, camera index, `RECOGNITION_THRESHOLD`, detector size, `DETECT_EVERY_N_FRAMES`). Every other module imports `config` rather than hard-coding values; change behavior here.
- **`face_engine.py`** — wraps InsightFace. The `FaceAnalysis` app is a lazily-built cached singleton (`get_app()`), and InsightFace/ONNX are imported *inside* that function so importing the module stays cheap. `cosine_match()` relies on embeddings being L2-normalized, so cosine similarity is computed as a plain dot product (`known_matrix @ emb`); do not change one without the other.
- **`database.py`** — SQLite layer with two tables: `users` and `face_encodings` (embeddings stored as float32 BLOBs, `ON DELETE CASCADE` from users). Each user can have multiple encodings. Uses a custom `_connect()` contextmanager that both commits *and* closes — sqlite3's built-in `with conn` only manages the transaction and leaks the handle, which locks the DB file on Windows. `load_all_encodings()` returns an `[N, 512]` matrix plus a parallel `user_ids` list; this is the in-memory index recognition matches against.
- **`pipeline.py`** — the shared logic layer that both frontends call, so enrollment/matching rules live in exactly one place. Holds `enroll_frame()` / `capture_single_face()` (the "exactly one face" enrollment rule), the `KnownFaces` index (`.load()` / `.identify()` / `.reload()`, wrapping `load_all_encodings` plus a user-info cache), the `Match` dataclass, and `draw_label()`. Depends only on the backend (`config`, `database`, `face_engine`) plus cv2/numpy — no GUI/CLI concerns.
- **`VideoCaptureService.py`** — owns the `cv2.VideoCapture` plus its own background detect loop, so the camera/threading lives in one place instead of in each frontend. Two daemon threads (a reader keeping the latest BGR frame, a detector running an injected `detect_fn(frame)` and keeping the latest result) feed thread-safe accessors `latest_frame()` / `latest_result()`. The `detect_fn` is swappable at runtime (`set_detect_fn`), with an internal epoch guard that drops a result computed by the previous callback so a renderer never sees the wrong kind. Depends only on `cv2`/`threading`/`config` — never on `face_engine`/`pipeline`, so it stays detection-agnostic. All three frontends drive it; supports `with` for the CLIs.
- **`enroll.py`** / **`recognize.py`** — the two CLI entry points; thin window loops that read `latest_frame()` / `latest_result()` from a `VideoCaptureService` and handle only drawing + keypresses. `recognize.py` passes `known.identify` as the `detect_fn`; `enroll.py` passes `face_engine.detect` for the live face boxes and enrolls the latest frame on SPACE. Detection now runs continuously on the service's own thread, so the old `DETECT_EVERY_N_FRAMES` frame-skip throttle is gone (the constant remains in `config.py` but is no longer consumed).
- **`app.py`** — Tkinter + Pillow desktop GUI: a tabbed window (Enroll / Recognize) sharing one `VideoCaptureService`. The UI thread renders `latest_frame()`/`latest_result()` via `root.after()`; the service's detect thread runs the heavy `pipeline`/`face_engine` work. On tab change it `set_detect_fn`s between the Enroll callback (face boxes) and the Recognize callback (`KnownFaces.identify`), `reload()`ing the index when entering Recognize so just-enrolled faces are recognized without a restart. Importing the module must not open the camera or load the model.

## Notes

- `users.csv` is sample data and is **not** read by the app.
- Recognition loads embeddings into memory at startup — newly enrolled faces require restarting `recognize.py`.
- For a CUDA GPU, swap `onnxruntime` for `onnxruntime-gpu` in `requirements.txt` (and see `CTX_ID` in `config.py`).
