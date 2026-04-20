"""
Voice pre-cloning cache for MOSS-TTS-Nano.

Encodes a reference audio file through the MOSS-Audio-Tokenizer-Nano once,
saves the resulting RVQ codes to disk, and restores them on subsequent calls
so that model.inference() skips the encoding step entirely.

Cache format  (torch.save / torch.load):
    {
        "audio_codes": Tensor[num_quantizers, 1, seq_len],  # raw batch_encode output
        "source_path": str,   # absolute path used to produce the codes
    }

The monkey-patch strategy:
    audio_tokenizer.batch_encode is temporarily replaced with a lambda that
    returns a SimpleNamespace(audio_codes=<cached_tensor>) — the same duck-type
    interface that model._call_audio_encode() expects.  All other methods
    (batch_decode, etc.) remain untouched on the tokenizer instance.
"""
from __future__ import annotations

import hashlib
import logging
import types
from pathlib import Path
from typing import Any, Optional

import torch
import torchaudio

from moss_tts_nano.config import get_cache_dir

logger = logging.getLogger(__name__)

# Known defaults for the MOSS-Audio-Tokenizer-Nano — read from tokenizer config
# at runtime when possible; these are fallbacks.
_DEFAULT_SAMPLE_RATE = 48_000
_DEFAULT_CHANNELS = 2


# ---------------------------------------------------------------------------
# Key / path helpers
# ---------------------------------------------------------------------------

def cache_key(audio_path: Path) -> str:
    """Return a stable SHA-256 hex key for *audio_path*."""
    canonical = str(audio_path.expanduser().resolve())
    return hashlib.sha256(canonical.encode()).hexdigest()


def cache_file_for(audio_path: Path) -> Path:
    """Return the `.pt` cache file path for *audio_path* (may not exist yet)."""
    return get_cache_dir() / f"{cache_key(audio_path)}.pt"


# ---------------------------------------------------------------------------
# Audio loading helpers (mirrors model._load_reference_audio)
# ---------------------------------------------------------------------------

def _resolve_tokenizer_audio_spec(audio_tokenizer) -> tuple[int, int]:
    """Return (sample_rate, num_channels) expected by *audio_tokenizer*."""
    cfg = getattr(audio_tokenizer, "config", None)
    sample_rate: int = _DEFAULT_SAMPLE_RATE
    channels: int = _DEFAULT_CHANNELS
    for holder in (audio_tokenizer, cfg):
        if holder is None:
            continue
        for attr in ("sample_rate", "sampling_rate"):
            v = getattr(holder, attr, None)
            if v is not None:
                sample_rate = int(v)
                break
        for attr in ("number_channels", "num_channels", "channels"):
            v = getattr(holder, attr, None)
            if v is not None:
                channels = int(v)
                break
    return sample_rate, channels


def _load_and_prepare_audio(
    audio_path: Path,
    target_sample_rate: int,
    target_channels: int,
    device: torch.device,
) -> torch.FloatTensor:
    """Load audio file and return a (C, T) float32 tensor ready for batch_encode."""
    waveform, sample_rate = torchaudio.load(str(audio_path))
    waveform = waveform.to(torch.float32)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    current_ch = int(waveform.shape[0])
    if current_ch != target_channels:
        if current_ch == 1 and target_channels > 1:
            waveform = waveform.repeat(target_channels, 1)
        elif current_ch > 1 and target_channels == 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        else:
            raise ValueError(
                f"Unsupported channel conversion: {current_ch} -> {target_channels}"
            )
    return waveform.to(device)


# ---------------------------------------------------------------------------
# Encoding & saving
# ---------------------------------------------------------------------------

def encode_and_save(
    audio_path: Path,
    audio_tokenizer_path: str,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Encode *audio_path* with the audio tokenizer and save to cache.

    Parameters
    ----------
    audio_path:
        Path to the reference wav/mp3 file.
    audio_tokenizer_path:
        HF repo-id or local directory for the audio tokenizer.
    device:
        Torch device to run encoding on.  Defaults to CPU.

    Returns
    -------
    The ``audio_codes`` tensor of shape ``(num_quantizers, 1, seq_len)``.
    """
    from transformers import AutoModel  # lazy import — keeps module lightweight

    if device is None:
        device = torch.device("cpu")

    audio_path = audio_path.expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {audio_path}")

    logger.info("Loading audio tokenizer from %s for pre-encoding …", audio_tokenizer_path)
    audio_tokenizer = AutoModel.from_pretrained(audio_tokenizer_path, trust_remote_code=True)
    audio_tokenizer.eval()
    audio_tokenizer = audio_tokenizer.to(device)

    target_sr, target_ch = _resolve_tokenizer_audio_spec(audio_tokenizer)
    logger.info(
        "Encoding reference audio %s  (target %d Hz, %d ch) …",
        audio_path,
        target_sr,
        target_ch,
    )

    waveform = _load_and_prepare_audio(audio_path, target_sr, target_ch, device)

    with torch.no_grad():
        encode_output = audio_tokenizer.batch_encode([waveform], chunk_duration=None)

    audio_codes: Optional[torch.Tensor] = getattr(encode_output, "audio_codes", None)
    if audio_codes is None:
        # Fallback: try common attribute names
        for attr in ("audio_token_ids", "codes", "tokens"):
            audio_codes = getattr(encode_output, attr, None)
            if audio_codes is not None:
                break
    if audio_codes is None:
        if torch.is_tensor(encode_output):
            audio_codes = encode_output
        elif isinstance(encode_output, (list, tuple)) and len(encode_output) > 0:
            audio_codes = encode_output[0]
    if audio_codes is None:
        raise RuntimeError(
            "Could not extract audio_codes from batch_encode output. "
            f"Output type: {type(encode_output)}"
        )

    # Ensure we detach and move to CPU for storage
    audio_codes = audio_codes.detach().cpu()

    out_path = cache_file_for(audio_path)
    torch.save(
        {
            "audio_codes": audio_codes,
            "source_path": str(audio_path),
        },
        out_path,
    )
    logger.info("Saved pre-encoded voice cache to %s", out_path)
    return audio_codes


# ---------------------------------------------------------------------------
# Loading & injecting cached codes
# ---------------------------------------------------------------------------

def load_cached_codes(audio_path: Path) -> Optional[Any]:
    """Load cached codes for *audio_path*, or return *None* on cache miss.

    Returns a ``SimpleNamespace`` with an ``audio_codes`` attribute — the same
    duck-type interface as ``MossAudioTokenizerEncoderOutput`` — ready to be
    returned from a monkey-patched ``batch_encode``.
    """
    path = cache_file_for(audio_path)
    if not path.exists():
        return None
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
        audio_codes = data["audio_codes"]
        logger.info(
            "Cache hit for %s  -> codes shape %s",
            audio_path.name,
            tuple(audio_codes.shape),
        )
        return types.SimpleNamespace(audio_codes=audio_codes)
    except Exception as exc:
        logger.warning("Failed to load voice cache from %s: %s", path, exc)
        return None


def patch_tokenizer_with_cache(audio_tokenizer, cached_output: Any) -> None:
    """Replace *audio_tokenizer*.batch_encode with a stub returning *cached_output*.

    The replacement is an instance-level attribute override so it only affects
    this specific tokenizer object; other instances are not changed.
    The tokenizer's ``batch_decode`` and all other methods remain intact.
    """
    def _cached_batch_encode(wav_list, **kwargs):  # noqa: ANN
        del wav_list, kwargs  # unused; we return the pre-computed codes
        return cached_output

    audio_tokenizer.batch_encode = _cached_batch_encode
    logger.info("Patched audio_tokenizer.batch_encode with cached voice codes.")
