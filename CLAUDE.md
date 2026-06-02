# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -r requirements.txt          # install deps (Python 3.8+, needs a webcam)

python -m facer.Enroll --name "Jane Doe" --email jane@example.com          # enroll from webcam (SPACE=capture, Q=finish)
python -m facer.Enroll --name "Jane Doe" --email jane@example.com --image jane.jpg   # enroll from a photo (exactly one face)

python -m facer.Recognize                 # run the always-on recognizer CLI (Q to quit)

python app.py                             # desktop GUI: enroll + recognize in one window
```

Run every command from the repo root: the library lives in the `facer/` package, but `config.py` and `app.py` stay at the root, so the root must be on `sys.path` (which it is when invoked from there). The CLIs use package-relative imports, hence `python -m facer.Enroll` rather than `python facer/Enroll.py`.

There is no build step, linter, or test suite configured. First run of either script downloads the `buffalo_l` model pack (~300 MB) to `~/.insightface`.

## Architecture

A layered pipeline around InsightFace embeddings stored in SQLite. Data flow: a BGR frame → `engine.detect()` (the shared `FaceEngine`) → 512-d L2-normalized embedding → cosine match against all stored embeddings → user lookup. Two frontends sit on top — the CLIs (`facer/Enroll.py`, `facer/Recognize.py`) and the desktop GUI (`app.py`) — and both go through the shared `facer/VideoCapturePipeline.py` layer so their behavior can't diverge.

The library modules live in the **`facer/`** package and import each other with package-relative imports (`from .Database import db`). `config.py` and `app.py` stay at the repo root, *outside* the package: package modules reach the settings via plain `import config` (resolved because the repo root is on `sys.path`), and `app.py` reaches the library via `from facer.X import Y`.

- **`config.py`** (root) — single source of all tunable settings (DB path, camera index, `RECOGNITION_THRESHOLD`, detector size, `DETECT_EVERY_N_FRAMES`). Every other module imports `config` rather than hard-coding values; change behavior here.
- **`facer/FaceEngine.py`** — the `FaceEngine` class wraps InsightFace. The `FaceAnalysis` app is built lazily and cached on the instance (`_get_app()`), with InsightFace/ONNX imported *inside* that method so importing the module stays cheap. A shared module-level singleton `engine = FaceEngine()` is what the rest of the package imports (`from .FaceEngine import engine`), so the ~300 MB model loads once. `FaceEngine.cosine_match()` (a staticmethod) relies on embeddings being L2-normalized, so cosine similarity is computed as a plain dot product (`known_matrix @ emb`); do not change one without the other.
- **`facer/Database.py`** — the `Database` class wraps the SQLite layer: two tables, `users` and `face_encodings` (embeddings stored as float32 BLOBs, `ON DELETE CASCADE` from users), and each user can have multiple encodings. The DB path is injected via the constructor (`Database(db_path=config.DB_PATH)`), and a shared module-level singleton `db = Database()` is what the rest of the package imports (`from .Database import db`), mirroring `FaceEngine`'s `engine`. Uses a custom `_connect()` contextmanager method that both commits *and* closes — sqlite3's built-in `with conn` only manages the transaction and leaks the handle, which locks the DB file on Windows. `db.load_all_encodings()` returns an `[N, 512]` matrix plus a parallel `user_ids` list; this is the in-memory index recognition matches against.
- **`facer/VideoCapturePipeline.py`** — the shared logic layer that both frontends call, so enrollment/matching rules live in exactly one place. The `VideoCapturePipeline` class holds the stateless helpers `enroll_frame()` / `capture_single_face()` (the "exactly one face" enrollment rule) and `draw_label()` as staticmethods, exposed app-wide through a `pipeline = VideoCapturePipeline()` singleton (`from .VideoCapturePipeline import pipeline`, mirroring `engine`/`db`). The file also holds the `KnownFaces` index (`.load()` / `.identify()` / `.reload()`, wrapping `load_all_encodings` plus a user-info cache) and the `Match` dataclass as sibling classes, plus the `COLOR_KNOWN` / `COLOR_UNKNOWN` module constants. Depends only on the backend (`config`, `Database`, `FaceEngine`) plus cv2/numpy — no GUI/CLI concerns.
- **`facer/VideoCaptureService.py`** — owns the `cv2.VideoCapture` plus its own background detect loop, so the camera/threading lives in one place instead of in each frontend. Two daemon threads (a reader keeping the latest BGR frame, a detector running an injected `detect_fn(frame)` and keeping the latest result) feed thread-safe accessors `latest_frame()` / `latest_result()`. The `detect_fn` is swappable at runtime (`set_detect_fn`), with an internal epoch guard that drops a result computed by the previous callback so a renderer never sees the wrong kind. Depends only on `cv2`/`threading`/`config` — never on `FaceEngine`/`VideoCapturePipeline`, so it stays detection-agnostic. All three frontends drive it; supports `with` for the CLIs.
- **`facer/Enroll.py`** / **`facer/Recognize.py`** — the two CLI entry points (run as `python -m facer.Enroll` / `python -m facer.Recognize`); thin window loops that read `latest_frame()` / `latest_result()` from a `VideoCaptureService` and handle only drawing + keypresses. `Enroll.py` wraps a session in the `Enroll` class (`__init__` creates the DB user; `from_image` / `from_camera` attach encodings) and passes `engine.detect` for the live face boxes, enrolling the latest frame on SPACE; `Recognize.py` wraps a session in the `Recognize` class (`__init__` loads the `KnownFaces` index; `run()` is the loop) and passes `known.identify` as the `detect_fn`. Detection now runs continuously on the service's own thread, so the old `DETECT_EVERY_N_FRAMES` frame-skip throttle is gone (the constant remains in `config.py` but is no longer consumed).
- **`app.py`** (root) — the desktop launcher only: `main()` builds the Tk root, shows the splash, then constructs `FacerApp`. No app logic lives here.
- **`facer/FacerApp.py`** — Tkinter + Pillow desktop GUI shell. `FacerApp` wires up the `VideoCaptureService` + `KnownFaces`, composes a `VideoNotebook`, and runs only the `root.after()` frame loop (hand the latest frame + result to `notebook.active_tab().render(...)`) plus shutdown. No widgets or per-tab logic live here. Importing the module must not open the camera or load the model.
- **`facer/VideoNotebook.py`** — wraps a `ttk.Notebook` holding the two tab classes (composition, like `FacerSplash`). On `<<NotebookTabChanged>>` it calls the new tab's `on_enter()` (which swaps the service's `detect_fn`); `active_tab()` returns the live tab for the frame loop to render. Home to the shared `to_photo()` (BGR→`PhotoImage`), which it injects into both tabs so they share it without importing this module back (would be a circular import).
- **`facer/VideoNotebookEnrollTab.py`** / **`facer/VideoNotebookRecognizeTab.py`** — one class per tab; each owns its widgets (on `self.frame`) and state, and exposes the trio the notebook drives: `detect(frame)` (run in the service's background thread — `engine.detect` boxes vs. `KnownFaces.identify`), `render(frame, result)` (draw into its own preview), and `on_enter()` (point the service at this tab's detection; Recognize also `reload()`s the index first). The Enroll tab also holds `_on_capture`/`_on_reset`, the Recognize tab `_on_capture_result`/`_set_result`.

## Notes

- `users.csv` is sample data and is **not** read by the app.
- Recognition loads embeddings into memory at startup — newly enrolled faces require restarting `python -m facer.Recognize`.
- For a CUDA GPU, swap `onnxruntime` for `onnxruntime-gpu` in `requirements.txt` (and see `CTX_ID` in `config.py`).
