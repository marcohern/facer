"""SQLite persistence layer for users and their face embeddings."""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np

import config


class Database:
    """SQLite persistence for users and their face embeddings.

    Two tables: `users` and `face_encodings` (embeddings stored as float32
    BLOBs, `ON DELETE CASCADE` from users). The DB path is injected at
    construction so the whole app can share one instance via the `db` singleton
    below.
    """

    def __init__(self, db_path=config.DB_PATH):
        self._db_path = db_path

    @contextmanager
    def _connect(self):
        """Open a connection that commits on success and always closes.

        sqlite3's own `with conn` only manages the transaction, not closing, which
        leaks the handle and keeps the DB file locked on Windows.
        """
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_db(self):
        """Create the tables if they do not already exist."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id    INTEGER PRIMARY KEY AUTOINCREMENT,
                    name  TEXT NOT NULL,
                    email TEXT UNIQUE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS face_encodings (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    embedding  BLOB NOT NULL,
                    created_at TEXT
                )
                """
            )

    def add_user(self, name, email):
        """Insert a user, or return the id of an existing user with the same email."""
        with self._connect() as conn:
            if email:
                row = conn.execute(
                    "SELECT id FROM users WHERE email = ?", (email,)
                ).fetchone()
                if row:
                    return row["id"]
            cur = conn.execute(
                "INSERT INTO users (name, email) VALUES (?, ?)", (name, email)
            )
            return cur.lastrowid

    def add_encoding(self, user_id, embedding):
        """Store one face embedding (np.ndarray) for a user as a float32 BLOB."""
        blob = np.asarray(embedding, dtype=np.float32).tobytes()
        created = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO face_encodings (user_id, embedding, created_at) "
                "VALUES (?, ?, ?)",
                (user_id, blob, created),
            )

    def count_encodings(self, user_id):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM face_encodings WHERE user_id = ?", (user_id,)
            ).fetchone()
            return row["n"]

    def get_user(self, user_id):
        """Return a user as a dict, or None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, name, email FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            return dict(row) if row else None

    def load_all_encodings(self):
        """Load every stored embedding.

        Returns (matrix, user_ids) where matrix is an [N, 512] float32 array and
        user_ids[i] is the user that embedding i belongs to. If there are no
        encodings, returns an empty (0, 0) array and an empty list.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT user_id, embedding FROM face_encodings"
            ).fetchall()

        if not rows:
            return np.empty((0, 0), dtype=np.float32), []

        user_ids = [r["user_id"] for r in rows]
        vectors = [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
        matrix = np.vstack(vectors)
        return matrix, user_ids


# Shared singleton: the whole app reuses one Database instance, mirroring
# FaceEngine's `engine`. Lazy — constructing it here does not touch the DB file.
db = Database()
