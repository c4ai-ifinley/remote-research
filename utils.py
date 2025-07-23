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


def create_empty_test_cases_config(config_path: str = "test_cases.json") -> bool:
    """
    Create an empty but valid test_cases.json configuration file.

    Args:
        config_path: Path to the config file to create

    Returns:
        bool: True if created successfully, False if file already exists
    """
    if os.path.exists(config_path):
        return False  # File already exists

    empty_config = {"test_cases": {}, "dspy_config": {"optimization_enabled": True}}

    atomic_write_json(empty_config, config_path)
    return True


def ensure_test_cases_config(config_path: str = "test_cases.json") -> None:
    """
    Ensure a valid test_cases.json configuration file exists.
    Creates one if missing or fixes it if empty/invalid.

    Args:
        config_path: Path to the config file to check/create
    """
    try:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                content = f.read().strip()
                if not content:
                    # File exists but is empty
                    create_empty_test_cases_config(config_path)
                    return

                # Check if valid JSON
                try:
                    config = json.loads(content)
                    # Ensure basic structure exists
                    if not isinstance(config, dict) or "test_cases" not in config:
                        create_empty_test_cases_config(config_path)
                except json.JSONDecodeError:
                    # Invalid JSON, recreate
                    create_empty_test_cases_config(config_path)
        else:
            # File doesn't exist, create it
            create_empty_test_cases_config(config_path)

    except Exception:
        # Any other error, try to create fresh file
        create_empty_test_cases_config(config_path)
