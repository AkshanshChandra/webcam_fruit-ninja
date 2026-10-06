"""Crash-safe JSON leaderboard.

Every write goes to a temp file which is fsync'd and atomically renamed over
leaderboard.json, so a crash or power-off mid-write can never leave a
half-written file. A corrupt file is moved aside (never silently deleted)."""

import json
import os
import tempfile
import time
import uuid

import config


class Leaderboard:
    def __init__(self, path=config.LEADERBOARD_FILE, keep=config.LEADERBOARD_KEEP):
        self.path = path
        self.keep = keep
        self.entries = []
        self.load()
        if not os.path.exists(self.path):
            self._write()

    def load(self):
        if not os.path.exists(self.path):
            self.entries = []
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data.get("scores", data) if isinstance(data, dict) else data
            self.entries = [self._clean(e) for e in entries if isinstance(e, dict)]
        except (OSError, ValueError) as exc:
            backup = f"{self.path}.corrupt-{int(time.time())}"
            print(f"[leaderboard] could not read {self.path} ({exc}); moved to {backup}")
            try:
                os.replace(self.path, backup)
            except OSError:
                pass
            self.entries = []
        self._sort()

    @staticmethod
    def _clean(e):
        return {
            "id": str(e.get("id") or uuid.uuid4().hex),
            "name": str(e.get("name", "PLAYER"))[:16] or "PLAYER",
            "score": int(e.get("score", 0)),
            "mode": str(e.get("mode", "survival")),
            "timestamp": str(e.get("timestamp", "")),
        }

    def _sort(self):
        # Highest score first; on ties the earlier entry ranks higher.
        self.entries.sort(key=lambda e: (-e["score"], e["timestamp"]))
        del self.entries[self.keep:]

    def _write(self):
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".leaderboard-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"scores": self.entries}, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def save(self):
        try:
            self._write()
            return True
        except OSError as exc:
            print(f"[leaderboard] save failed: {exc}")
            return False

    def best(self):
        return self.entries[0]["score"] if self.entries else 0

    def top(self, n=config.LEADERBOARD_SHOW):
        return self.entries[:n]

    def qualifies(self, score):
        return score > 0 and (len(self.entries) < self.keep or score > self.entries[-1]["score"])

    def add(self, name, score, mode):
        """Insert and persist immediately. Returns the entry id (or None if it
        didn't make the kept list)."""
        entry = {
            "id": uuid.uuid4().hex,
            "name": (name.strip() or "PLAYER")[:16],
            "score": int(score),
            "mode": mode,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.entries.append(entry)
        self._sort()
        self.save()
        return entry["id"] if any(e["id"] == entry["id"] for e in self.entries) else None

    def rename(self, entry_id, name):
        for e in self.entries:
            if e["id"] == entry_id:
                e["name"] = (name.strip() or "PLAYER")[:16]
                self.save()
                return True
        return False

    def remove(self, entry_id):
        before = len(self.entries)
        self.entries = [e for e in self.entries if e["id"] != entry_id]
        if len(self.entries) != before:
            self.save()

    def rank_of(self, entry_id):
        for i, e in enumerate(self.entries):
            if e["id"] == entry_id:
                return i + 1
        return None
