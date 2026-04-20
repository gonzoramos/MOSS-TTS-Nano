from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

from moss_tts_nano.defaults import (
    DEFAULT_AUDIO_TOKENIZER_PATH,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_OUTPUT_DIR,
)

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moss-tts-nano",
        description="Command line interface for MOSS-TTS-Nano.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate audio by forwarding to infer.py.",
    )
    generate_parser.add_argument(
        "--backend",
        default="pytorch",
        choices=("pytorch", "onnx"),
        help="Inference backend to use.",
    )
    generate_parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="PyTorch backend only. MOSS-TTS-Nano checkpoint source. Can be a HF repo id or local directory.",
    )
    generate_parser.add_argument(
        "--audio-tokenizer",
        "--audio_tokenizer",
        dest="audio_tokenizer",
        default=str(DEFAULT_AUDIO_TOKENIZER_PATH),
        help="PyTorch backend only. MOSS-Audio-Tokenizer-Nano source. Can be a HF repo id or local directory.",
    )
    generate_parser.add_argument(
        "--onnx-model-dir",
        default=None,
        help="ONNX backend only. browser_onnx model directory. Defaults to the repo's auto-download location.",
    )
    generate_parser.add_argument(
        "--output",
        "--output-audio-path",
        dest="output_audio_path",
        default=str(DEFAULT_OUTPUT_DIR / "moss_tts_nano_output.wav"),
        help="Output wav path.",
    )
    text_group = generate_parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", help="Text to synthesize.")
    text_group.add_argument("--text-file", help="Path to a UTF-8 text file to synthesize.")
    generate_parser.add_argument(
        "--mode",
        default="voice_clone",
        choices=("voice_clone", "continuation"),
        help="Inference mode.",
    )
    generate_parser.add_argument(
        "--prompt-speech",
        "--prompt_speech",
        "--prompt-audio-path",
        dest="prompt_audio_path",
        default=None,
        help=(
            "Reference speech used for voice cloning.  "
            "If omitted the default voice set via 'set-voice' is used."
        ),
    )
    generate_parser.add_argument(
        "--prompt-text",
        default=None,
        help="PyTorch backend only. Reference transcript used by continuation mode.",
    )
    generate_parser.add_argument(
        "--voice",
        default="Junhao",
        help="ONNX backend only. Built-in voice preset used when --prompt-speech is not provided.",
    )
    generate_parser.add_argument("--device", default="auto", help="Execution device, e.g. auto/cpu/cuda.")
    generate_parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="Weights dtype.",
    )
    generate_parser.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
        help="ONNX backend only. onnxruntime intra-op thread count.",
    )
    generate_parser.add_argument("--max-new-frames", type=int, default=375)
    generate_parser.add_argument("--voice-clone-max-text-tokens", type=int, default=75)
    generate_parser.add_argument(
        "--sample-mode",
        default="fixed",
        choices=("greedy", "fixed", "full"),
        help="ONNX backend only. Sampling mode.",
    )
    generate_parser.add_argument(
        "--realtime-streaming-decode",
        type=int,
        nargs="?",
        const=1,
        default=1,
        choices=[0, 1],
        help="ONNX backend only. Use codec streaming decode path internally instead of full decode.",
    )
    generate_parser.add_argument("--text-temperature", type=float, default=1.0)
    generate_parser.add_argument("--text-top-p", type=float, default=1.0)
    generate_parser.add_argument("--text-top-k", type=int, default=50)
    generate_parser.add_argument("--audio-temperature", type=float, default=0.8)
    generate_parser.add_argument("--audio-top-p", type=float, default=0.95)
    generate_parser.add_argument("--audio-top-k", type=int, default=25)
    generate_parser.add_argument("--audio-repetition-penalty", type=float, default=1.2)
    generate_parser.add_argument("--seed", type=int, default=None)
    generate_parser.add_argument(
        "--enable-wetext-processing",
        action="store_true",
        help="Enable WeTextProcessing normalization before inference.",
    )
    generate_parser.add_argument(
        "--print-voice-clone-text-chunks",
        action="store_true",
        help="Print sentence chunks before generation.",
    )
    generate_parser.set_defaults(handler=_run_generate)

    # ------------------------------------------------------------------
    # serve
    # ------------------------------------------------------------------
    serve_parser = subparsers.add_parser(
        "serve",
        help="Launch the FastAPI demo by forwarding to app.py.",
    )
    serve_parser.add_argument(
        "--backend",
        default="pytorch",
        choices=("pytorch", "onnx"),
        help="Inference backend to use.",
    )
    serve_parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="PyTorch backend only. MOSS-TTS-Nano checkpoint source. Can be a HF repo id or local directory.",
    )
    serve_parser.add_argument(
        "--audio-tokenizer",
        "--audio_tokenizer",
        dest="audio_tokenizer",
        default=str(DEFAULT_AUDIO_TOKENIZER_PATH),
        help="PyTorch backend only. MOSS-Audio-Tokenizer-Nano source. Can be a HF repo id or local directory.",
    )
    serve_parser.add_argument(
        "--onnx-model-dir",
        default=None,
        help="ONNX backend only. browser_onnx model directory. Defaults to the repo's auto-download location.",
    )
    serve_parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated wav files.",
    )
    serve_parser.add_argument("--device", default="auto", help="Execution device, e.g. auto/cpu/cuda.")
    serve_parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="Weights dtype.",
    )
    serve_parser.add_argument(
        "--attn-implementation",
        default="auto",
        choices=("auto", "sdpa", "flash_attention_2", "eager"),
        help="Attention backend override.",
    )
    serve_parser.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
        help="ONNX backend only. onnxruntime intra-op thread count.",
    )
    serve_parser.add_argument(
        "--max-new-frames",
        type=int,
        default=375,
        help="ONNX backend only. Maximum generated audio frames.",
    )
    serve_parser.add_argument("--host", default="localhost")
    serve_parser.add_argument("--port", type=int, default=18083)
    serve_parser.add_argument("--share", action="store_true", help="Accepted for compatibility.")
    serve_parser.set_defaults(handler=_run_serve)

    # ------------------------------------------------------------------
    # set-voice
    # ------------------------------------------------------------------
    sv_parser = subparsers.add_parser(
        "set-voice",
        help="Configure a default reference voice so --prompt-speech is optional.",
    )
    sv_source = sv_parser.add_mutually_exclusive_group()
    sv_source.add_argument(
        "--prompt-speech",
        "--prompt-audio-path",
        dest="prompt_audio_path",
        metavar="AUDIO_FILE",
        help="Path to a wav/mp3 file to use as the default voice.",
    )
    sv_source.add_argument(
        "--voice",
        metavar="PRESET_NAME",
        help="Name of a built-in voice preset (e.g. Junhao, Trump, Sakura).",
    )
    sv_parser.add_argument(
        "--audio-tokenizer",
        "--audio_tokenizer",
        dest="audio_tokenizer",
        default=str(DEFAULT_AUDIO_TOKENIZER_PATH),
        help="Audio tokenizer used for pre-encoding.  Defaults to the standard HF repo.",
    )
    sv_parser.add_argument(
        "--device",
        default="cpu",
        help="Device for pre-encoding (cpu or cuda).  Defaults to cpu.",
    )
    sv_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Save the voice path to config but skip pre-encoding the RVQ codes.",
    )
    sv_parser.add_argument(
        "--show",
        action="store_true",
        help="Print the current default voice config and exit.",
    )
    sv_parser.add_argument(
        "--clear",
        action="store_true",
        help="Remove the default voice setting.",
    )
    sv_parser.set_defaults(handler=_run_set_voice)

    return parser


# ---------------------------------------------------------------------------
# generate handler
# ---------------------------------------------------------------------------

def _run_generate_pytorch(args: argparse.Namespace) -> int:
    import infer as infer_module
    from moss_tts_nano import config as cfg_mod
    from moss_tts_nano import voice_cache

    prompt_audio_path: Optional[str] = args.prompt_audio_path
    preloaded_audio_tokenizer = None

    # --- fall back to the configured default voice if none was supplied -----
    if not prompt_audio_path:
        default_voice = cfg_mod.get_default_voice()
        if default_voice:
            prompt_audio_path = default_voice.get("path")
            logger.info("Using default voice: %s", prompt_audio_path)

            # Try to inject pre-encoded codes to skip the encoding step.
            cache_path_str = default_voice.get("cache_path")
            tokenizer_path = default_voice.get("audio_tokenizer_path") or str(DEFAULT_AUDIO_TOKENIZER_PATH)
            if cache_path_str and Path(cache_path_str).exists() and prompt_audio_path:
                cached = voice_cache.load_cached_codes(Path(prompt_audio_path))
                if cached is not None:
                    preloaded_audio_tokenizer = _load_and_patch_tokenizer(
                        tokenizer_path=tokenizer_path,
                        cached_output=cached,
                    )

    infer_argv: list[str] = [
        "--checkpoint",
        str(args.checkpoint),
        "--output-audio-path",
        str(args.output_audio_path),
        "--mode",
        args.mode,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--max-new-frames",
        str(args.max_new_frames),
        "--voice-clone-max-text-tokens",
        str(args.voice_clone_max_text_tokens),
        "--audio-tokenizer-pretrained-name-or-path",
        str(args.audio_tokenizer),
        "--text-temperature",
        str(args.text_temperature),
        "--text-top-p",
        str(args.text_top_p),
        "--text-top-k",
        str(args.text_top_k),
        "--audio-temperature",
        str(args.audio_temperature),
        "--audio-top-p",
        str(args.audio_top_p),
        "--audio-top-k",
        str(args.audio_top_k),
        "--audio-repetition-penalty",
        str(args.audio_repetition_penalty),
    ]
    if args.text is not None:
        infer_argv.extend(["--text", args.text])
    if args.text_file is not None:
        infer_argv.extend(["--text-file", args.text_file])
    if args.prompt_text:
        infer_argv.extend(["--prompt-text", args.prompt_text])
    if prompt_audio_path:
        infer_argv.extend(["--prompt-audio-path", prompt_audio_path])
    if args.seed is not None:
        infer_argv.extend(["--seed", str(args.seed)])
    if args.enable_wetext_processing:
        infer_argv.append("--enable-wetext-processing")
    else:
        infer_argv.append("--disable-wetext-processing")
    if args.print_voice_clone_text_chunks:
        infer_argv.append("--print-voice-clone-text-chunks")

    infer_module.main(infer_argv, audio_tokenizer=preloaded_audio_tokenizer)
    return 0


def _validate_onnx_generate_args(args: argparse.Namespace) -> None:
    if args.mode != "voice_clone":
        raise SystemExit("The ONNX backend currently supports only `--mode voice_clone`.")
    if args.prompt_text:
        raise SystemExit("The ONNX backend does not support `--prompt-text` yet.")
    if args.device not in {"auto", "cpu"}:
        raise SystemExit("The ONNX backend is CPU-only. Use `--device auto` or `--device cpu`.")
    if args.dtype not in {"auto", "float32"}:
        raise SystemExit("The ONNX backend supports only `--dtype auto` or `--dtype float32`.")


def _run_generate_onnx(args: argparse.Namespace) -> int:
    import infer_onnx as infer_onnx_module

    _validate_onnx_generate_args(args)
    infer_argv: list[str] = [
        "--output-audio-path",
        str(args.output_audio_path),
        "--voice",
        str(args.voice),
        "--sample-mode",
        str(args.sample_mode),
        "--do-sample",
        "0" if str(args.sample_mode) == "greedy" else "1",
        "--realtime-streaming-decode",
        str(int(bool(args.realtime_streaming_decode))),
        "--cpu-threads",
        str(args.cpu_threads),
        "--max-new-frames",
        str(args.max_new_frames),
        "--voice-clone-max-text-tokens",
        str(args.voice_clone_max_text_tokens),
        "--text-temperature",
        str(args.text_temperature),
        "--text-top-p",
        str(args.text_top_p),
        "--text-top-k",
        str(args.text_top_k),
        "--audio-temperature",
        str(args.audio_temperature),
        "--audio-top-p",
        str(args.audio_top_p),
        "--audio-top-k",
        str(args.audio_top_k),
        "--audio-repetition-penalty",
        str(args.audio_repetition_penalty),
    ]
    if args.onnx_model_dir:
        infer_argv.extend(["--model-dir", str(args.onnx_model_dir)])
    if args.text is not None:
        infer_argv.extend(["--text", args.text])
    if args.text_file is not None:
        infer_argv.extend(["--text-file", args.text_file])
    if args.prompt_audio_path:
        infer_argv.extend(["--prompt-audio-path", args.prompt_audio_path])
    if args.seed is not None:
        infer_argv.extend(["--seed", str(args.seed)])
    if args.enable_wetext_processing:
        infer_argv.append("--enable-wetext-processing")
    else:
        infer_argv.append("--disable-wetext-processing")
    if args.print_voice_clone_text_chunks:
        infer_argv.append("--print-voice-clone-text-chunks")
    infer_onnx_module.main(infer_argv)
    return 0


def _run_generate(args: argparse.Namespace) -> int:
    if str(args.backend) == "onnx":
        return _run_generate_onnx(args)
    return _run_generate_pytorch(args)


def _load_and_patch_tokenizer(tokenizer_path: str, cached_output):
    """Load the audio tokenizer and monkey-patch batch_encode with cached codes."""
    from transformers import AutoModel
    from moss_tts_nano import voice_cache

    logger.info("Loading audio tokenizer (%s) to inject cached voice codes …", tokenizer_path)
    audio_tokenizer = AutoModel.from_pretrained(tokenizer_path, trust_remote_code=True)
    audio_tokenizer.eval()
    voice_cache.patch_tokenizer_with_cache(audio_tokenizer, cached_output)
    return audio_tokenizer


# ---------------------------------------------------------------------------
# set-voice handler
# ---------------------------------------------------------------------------

def _run_set_voice(args: argparse.Namespace) -> int:
    import json
    from moss_tts_nano import config as cfg_mod
    from moss_tts_nano import voice_cache

    # -- show ---------------------------------------------------------------
    if args.show:
        dv = cfg_mod.get_default_voice()
        if dv is None:
            print("No default voice configured.")
        else:
            print(json.dumps(dv, indent=2))
        return 0

    # -- clear --------------------------------------------------------------
    if args.clear:
        cfg_mod.clear_default_voice()
        print("Default voice cleared.")
        return 0

    # -- resolve audio path -------------------------------------------------
    audio_path: Optional[Path] = None
    voice_type = "file"

    if args.prompt_audio_path:
        audio_path = Path(args.prompt_audio_path).expanduser().resolve()
        if not audio_path.exists():
            print(f"Error: file not found: {audio_path}")
            return 1
        voice_type = "file"

    elif args.voice:
        audio_path = _resolve_preset_path(args.voice)
        if audio_path is None:
            print(f"Error: unknown preset '{args.voice}'.")
            print("Run 'moss-tts-nano set-voice --show' or check assets/audio/ for available files.")
            return 1
        voice_type = "preset"

    else:
        print("Error: specify --prompt-speech <file> or --voice <name>.  Use --show to inspect current config.")
        return 1

    # -- pre-encode ---------------------------------------------------------
    cache_path_str: Optional[str] = None
    if not args.no_cache:
        try:
            import torch
            device = torch.device(args.device)
            voice_cache.encode_and_save(
                audio_path=audio_path,
                audio_tokenizer_path=args.audio_tokenizer,
                device=device,
            )
            cache_path_str = str(voice_cache.cache_file_for(audio_path))
            print(f"Pre-encoded voice saved to: {cache_path_str}")
        except Exception as exc:
            print(f"Warning: pre-encoding failed ({exc}). Saving voice path only.")
            logger.debug("Pre-encoding error", exc_info=True)

    # -- persist config -----------------------------------------------------
    cfg_mod.set_default_voice(
        voice_type=voice_type,
        audio_path=str(audio_path),
        cache_path=cache_path_str,
        audio_tokenizer_path=args.audio_tokenizer,
    )
    print(f"Default voice set to: {audio_path}")
    if cache_path_str:
        print("RVQ codes pre-encoded — subsequent 'generate' calls will skip the encoding step.")
    else:
        print("No RVQ cache. The encoding step will run on each 'generate' call as usual.")
    return 0


def _resolve_preset_path(voice_name: str) -> Optional[Path]:
    """Return the absolute audio path for a named voice preset, or None."""
    try:
        from moss_tts_nano_runtime import NanoTTSService  # lazy heavy import
        dummy = object.__new__(NanoTTSService)  # skip __init__
        preset_map = getattr(NanoTTSService, "_DEFAULT_VOICE_FILES", None)
        if preset_map and voice_name in preset_map:
            filename, _ = preset_map[voice_name]
            from moss_tts_nano.defaults import DEFAULT_PROMPT_AUDIO_DIR
            p = DEFAULT_PROMPT_AUDIO_DIR / filename
            if p.exists():
                return p.resolve()
        from moss_tts_nano_runtime import NanoTTSService as SVC
        tmp = SVC.__new__(SVC)
        tmp._voice_presets = SVC._build_default_voice_presets() if hasattr(SVC, "_build_default_voice_presets") else {}
        if hasattr(tmp, "get_voice_preset"):
            preset = tmp.get_voice_preset(voice_name)
            if preset and hasattr(preset, "prompt_audio_path"):
                p = Path(preset.prompt_audio_path)
                if p.exists():
                    return p.resolve()
        del dummy, tmp
    except Exception:
        pass
    from moss_tts_nano.defaults import DEFAULT_PROMPT_AUDIO_DIR
    for ext in (".wav", ".mp3"):
        candidate = DEFAULT_PROMPT_AUDIO_DIR / f"{voice_name}{ext}"
        if candidate.exists():
            return candidate.resolve()
    return None


# ---------------------------------------------------------------------------
# serve handler
# ---------------------------------------------------------------------------

def _validate_onnx_serve_args(args: argparse.Namespace) -> None:
    if args.device not in {"auto", "cpu"}:
        raise SystemExit("The ONNX backend is CPU-only. Use `--device auto` or `--device cpu`.")
    if args.dtype not in {"auto", "float32"}:
        raise SystemExit("The ONNX backend supports only `--dtype auto` or `--dtype float32`.")
    if args.attn_implementation != "auto":
        raise SystemExit("The ONNX backend does not use `--attn-implementation`; leave it as `auto`.")


def _run_serve_pytorch(args: argparse.Namespace) -> int:
    import app as app_module

    app_argv = [
        "--checkpoint-path",
        str(args.checkpoint),
        "--audio-tokenizer-path",
        str(args.audio_tokenizer),
        "--output-dir",
        str(args.output_dir),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--attn-implementation",
        args.attn_implementation,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.share:
        app_argv.append("--share")
    app_module.main(app_argv)
    return 0


def _run_serve_onnx(args: argparse.Namespace) -> int:
    import app_onnx as app_onnx_module

    _validate_onnx_serve_args(args)
    app_argv = [
        "--output-dir",
        str(args.output_dir),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--cpu-threads",
        str(args.cpu_threads),
        "--max-new-frames",
        str(args.max_new_frames),
    ]
    if args.onnx_model_dir:
        app_argv.extend(["--model-dir", str(args.onnx_model_dir)])
    if args.share:
        app_argv.append("--share")
    app_onnx_module.main(app_argv)
    return 0


def _run_serve(args: argparse.Namespace) -> int:
    if str(args.backend) == "onnx":
        return _run_serve_onnx(args)
    return _run_serve_pytorch(args)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))
