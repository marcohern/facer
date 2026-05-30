"""Central configuration for the face recognition app."""

# Path to the SQLite database file.
DB_PATH = "faces.db"

# Index of the camera to open (0 = default webcam).
CAMERA_INDEX = 0

# Cosine-similarity threshold for treating a face as a match.
# InsightFace buffalo_l embeddings are L2-normalized, so cosine == dot product.
# Higher = stricter. ~0.5 is a sensible default; tune for your camera/lighting.
RECOGNITION_THRESHOLD = 0.5

# InsightFace detector input size (width, height). Larger = more accurate, slower.
DET_SIZE = (640, 640)

# Model pack name downloaded automatically by InsightFace on first run.
MODEL_NAME = "buffalo_l"

# ctx_id for InsightFace: 0 uses the first available provider (GPU if present,
# otherwise CPU). Set to -1 to force CPU.
CTX_ID = 0

# Run heavy detection/recognition only every Nth frame to keep the window smooth.
# Frames in between reuse the last results.
DETECT_EVERY_N_FRAMES = 3
