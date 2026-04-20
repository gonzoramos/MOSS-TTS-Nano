"""
User-level configuration for MOSS-TTS-Nano.

Config is stored in  ~/.config/moss-tts-nano/config.json
Cache files live in  ~/.config/moss-tts-nano/cache/
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


_CONFIG_DIR_ENV = "MOSS_TTS_NANO_CONFIG_DIR"
_CONFIG_FILE = "config.json"
_CACHE_SUBDIR = "cache"


def get_config_dir() -> Path:
    """Return the configuration directory, creating it if needed."""
    env = os.environ.get(_CONFIG_DIR_ENV)
    base = Path(env) if env else Path.home() / ".config" / "moss-tts-nano"
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_cache_dir() -> Path:
    """Return the cache sub-directory, creating it if needed."""
    cache = get_config_dir() / _CACHE_SUBDIR
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _config_path() -> Path:
    return get_config_dir() / _CONFIG_FILE


def load_config() -> dict[str, Any]:
    """Load and return the full config dict.  Returns {} if the file does not exist."""
    path = _config_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_config(data: dict[str, Any]) -> None:
    """Atomically write *data* to the config file."""
    path = _config_path()
    tmp_fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def get_default_voice() -> Optional[dict[str, Any]]:
    """Return the ``default_voice`` block from config, or *None* if unset."""
    return load_config().get("default_voice")


def set_default_voice(
    *,
    voice_type: str,
    audio_path: str,
    cache_path: Optional[str] = None,
    audio_tokenizer_path: Optional[str] = None,
) -> None:
    """Persist the default voice settings.

    Parameters
    ----------
    voice_type:
        ``"file"`` for an explicit audio path, ``"preset"`` for a named preset.
    audio_path:
        Absolute path to the reference audio file.
    cache_path:
        Absolute path to the pre-encoded ``.pt`` cache file, or *None* if no
        cache was produced.
    audio_tokenizer_path:
        The audio tokenizer repo-id / local path used during encoding.
    """
    cfg = load_config()
    cfg["default_voice"] = {
        "type": voice_type,
        "path": audio_path,
        "cache_path": cache_path,
        "audio_tokenizer_path": audio_tokenizer_path,
        "cached_at": datetime.now(timezone.utc).isoformat(),
    }
    save_config(cfg)


def clear_default_voice() -> None:
    """Remove the default_voice entry from config."""
    cfg = load_config()
    cfg.pop("default_voice", None)
    save_config(cfg)
