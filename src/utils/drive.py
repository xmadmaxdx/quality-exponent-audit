"""Colab Drive autosave. No-op locally. Never fails a run."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

LOGGER_NAME = "quality-exponent-audit"

log = logging.getLogger(LOGGER_NAME)

_MOUNT = Path("/content/drive")
_OUT = _MOUNT / "MyDrive" / "quality-exponent-audit-runs"


def drive_mounted() -> bool:
    return _OUT.parent.is_dir()


def ensure_drive_mounted() -> bool:
    """Mount Drive on Colab if possible. Returns True when usable."""
    try:
        if drive_mounted():
            return True
        import google.colab  # noqa: F401
        from google.colab import drive

        drive.mount(str(_MOUNT), force_remount=False)
        _OUT.mkdir(parents=True, exist_ok=True)
        log.info("drive mounted at %s", _OUT)
        return True
    except Exception as exc:
        log.warning("drive not available, local-only mode: %s", exc)
        return False


def mirror_file(local: str | Path) -> Path | None:
    """Copy one file into the Drive mirror, preserving its parent folder name."""
    try:
        src = Path(local)
        if not src.is_file():
            return None
        if not drive_mounted():
            return None
        dest = _OUT / src.parent.name / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return dest
    except Exception as exc:
        log.warning("drive mirror failed for %s: %s", local, exc)
        return None