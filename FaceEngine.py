"""InsightFace wrapper: face detection, embeddings, and cosine matching."""

import numpy as np

import config


class FaceEngine:
    """Wraps an InsightFace `FaceAnalysis` app: built lazily, then reused.

    The heavy InsightFace/ONNX import and the ~300 MB model load happen on the
    first `detect()` call — not at construction — so creating a `FaceEngine`
    (or importing this module) stays cheap.
    """

    def __init__(self, model_name=config.MODEL_NAME, ctx_id=config.CTX_ID,
                 det_size=config.DET_SIZE):
        self._model_name = model_name
        self._ctx_id = ctx_id
        self._det_size = det_size
        self._app = None

    def _get_app(self):
        """Lazily build and cache the InsightFace FaceAnalysis app."""
        if self._app is None:
            # Imported here so simply importing this module (e.g. for tests)
            # does not pull in the heavy InsightFace/ONNX stack until it's used.
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(
                name=self._model_name, providers=['CPUExecutionProvider']
            )
            app.prepare(ctx_id=self._ctx_id, det_size=self._det_size)
            self._app = app
        return self._app

    def detect(self, frame_bgr):
        """Detect faces in a BGR frame.

        Returns a list of InsightFace Face objects. Each has, among others:
          - .bbox            -> np.ndarray [x1, y1, x2, y2]
          - .normed_embedding -> L2-normalized 512-d embedding
          - .det_score       -> detection confidence
        """
        return self._get_app().get(frame_bgr)

    @staticmethod
    def cosine_match(embedding, known_matrix, user_ids, threshold):
        """Match one embedding against a matrix of known embeddings.

        Embeddings are L2-normalized, so cosine similarity is just the dot
        product. Returns (user_id, score). user_id is None if there are no known
        faces or the best score is below `threshold`.
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


# Shared singleton: the model is heavy (~300 MB), so the whole app reuses one
# engine. Lazy — constructing it here does not load the model.
engine = FaceEngine()
