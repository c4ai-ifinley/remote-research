import json
import os
import tempfile
from pathlib import Path


def atomic_write_json(data, path: str) -> None:
    """Write JSON data atomically to the given path."""
    path_obj = Path(path)
    tmp_dir = path_obj.parent
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=tmp_dir, suffix=".tmp"
    ) as tmp_file:
        json.dump(data, tmp_file, indent=2)
        tmp_name = tmp_file.name
    os.replace(tmp_name, path_obj)
