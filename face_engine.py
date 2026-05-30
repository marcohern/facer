"""InsightFace wrapper: face detection, embeddings, and cosine matching."""

import numpy as np

import config

_app = None


def get_app():
    """Lazily build and cache the InsightFace FaceAnalysis app (singleton)."""
    global _app
    if _app is None:
        # Imported here so simply importing this module (e.g. for tests) does
        # not pull in the heavy InsightFace/ONNX stack until it's needed.
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name=config.MODEL_NAME)
        app.prepare(ctx_id=config.CTX_ID, det_size=config.DET_SIZE)
        _app = app
    return _app


def detect(frame_bgr):
    """Detect faces in a BGR frame.

    Returns a list of InsightFace Face objects. Each has, among others:
      - .bbox            -> np.ndarray [x1, y1, x2, y2]
      - .normed_embedding -> L2-normalized 512-d embedding
      - .det_score       -> detection confidence
    """
    return get_app().get(frame_bgr)


def cosine_match(embedding, known_matrix, user_ids, threshold):
    """Match one embedding against a matrix of known embeddings.

    Embeddings are L2-normalized, so cosine similarity is just the dot product.
    Returns (user_id, score). user_id is None if there are no known faces or
    the best score is below `threshold`.
    """
    if known_matrix is None or known_matrix.size == 0:
        return None, 0.0

    emb = np.asarray(embedding, dtype=np.float32)
    scores = known_matrix @ emb  # [N]
    best = int(np.argmax(scores))
    best_score = float(scores[best])

    if best_score >= threshold:
        return user_ids[best], best_score
    return None, best_score
