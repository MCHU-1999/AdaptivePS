import time
import json
import os


class Timer:
    """Context manager that measures elapsed time."""
    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self._start


def load_runtime_json(path: str) -> dict:
    """Load an existing runtime JSON, or return an empty dict if it doesn't exist."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}


def save_runtime_json(path: str, data: dict):
    """Deep-merge data into an existing JSON file (or create one) and save."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing = load_runtime_json(path)
    for key, value in data.items():
        if key in existing and isinstance(existing[key], dict) and isinstance(value, dict):
            existing[key].update(value)   # deep merge for nested dicts (e.g. metric3dv2_s + total_s)
        else:
            existing[key] = value
    with open(path, 'w') as f:
        json.dump(existing, f, indent=2)
