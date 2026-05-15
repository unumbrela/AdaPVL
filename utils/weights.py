from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch
from huggingface_hub import hf_hub_download, snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_ROOT = PROJECT_ROOT / "weights"


def project_root() -> Path:
    return PROJECT_ROOT


def weights_root() -> Path:
    WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)
    return WEIGHTS_ROOT


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_name(name: str) -> str:
    safe = name.replace("\\", "/")
    for old, new in (("/", "__"), (":", "__"), ("@", "__"), (" ", "_")):
        safe = safe.replace(old, new)
    return safe


def local_weight_dir(*parts: str) -> Path:
    return ensure_dir(weights_root().joinpath(*parts))


def local_weight_file(*parts: str) -> Path:
    path = weights_root().joinpath(*parts)
    ensure_dir(path.parent)
    return path


def _unique_paths(paths: Iterable[Path | str | None]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        if not path:
            continue
        resolved = Path(path).expanduser()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)
    return unique


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_valid_file(path: Path, expected_sha256: Optional[str] = None) -> bool:
    if not path.is_file():
        return False
    if not expected_sha256:
        return True
    return _sha256(path) == expected_sha256


def _copy_file(source: Path, target: Path) -> Path:
    ensure_dir(target.parent)
    if source.resolve() == target.resolve():
        return target
    shutil.copy2(source, target)
    return target


def _copy_tree(source: Path, target: Path) -> Path:
    ensure_dir(target.parent)
    shutil.copytree(source, target, dirs_exist_ok=True)
    return target


def _has_required_files(
    directory: Path,
    required_paths: Optional[Sequence[str]] = None,
    required_any_paths: Optional[Sequence[str]] = None,
) -> bool:
    if not directory.is_dir():
        return False
    if required_paths and not all((directory / rel_path).exists() for rel_path in required_paths):
        return False
    if required_any_paths and not any((directory / rel_path).exists() for rel_path in required_any_paths):
        return False
    return True


def _download_to_file(url: str, target: Path) -> Path:
    ensure_dir(target.parent)
    tmp_target = target.with_suffix(target.suffix + ".tmp")
    with urllib.request.urlopen(url) as source, tmp_target.open("wb") as handle:
        shutil.copyfileobj(source, handle)
    tmp_target.replace(target)
    return target


def _torch_load_cpu(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_timm_state_dict(source: Path) -> Optional[dict]:
    if source.is_dir():
        for name in (
            "model.safetensors",
            "pytorch_model.bin",
            "pytorch_model.pth",
            "model.pth",
        ):
            candidate = source / name
            if candidate.is_file():
                source = candidate
                break
        else:
            return None

    if source.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(
                f"safetensors is required to load timm weights from {source}"
            ) from exc
        return load_file(str(source), device="cpu")

    return _torch_load_cpu(source)


def clip_cache_dirs() -> list[Path]:
    return _unique_paths([
        Path(os.environ.get("XDG_CACHE_HOME", "")).expanduser() / "clip" if os.environ.get("XDG_CACHE_HOME") else None,
        Path.home() / ".cache" / "clip",
    ])


def hf_cache_dirs() -> list[Path]:
    env_hf_home = os.environ.get("HF_HOME")
    env_transformers_cache = os.environ.get("TRANSFORMERS_CACHE")
    env_hf_hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    return _unique_paths([
        env_transformers_cache,
        env_hf_hub_cache,
        env_hf_home,
        Path(env_hf_home).expanduser() / "hub" if env_hf_home else None,
        Path(xdg_cache_home).expanduser() / "huggingface" if xdg_cache_home else None,
        Path.home() / ".cache" / "huggingface",
        Path.home() / ".cache" / "huggingface" / "hub",
    ])


def timm_url_cache_dirs() -> list[Path]:
    torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")).expanduser()
    return _unique_paths([
        torch_home / "hub" / "checkpoints",
    ])


def resolve_url_weight(
    url: str,
    local_subdir: str | Path,
    *,
    cache_dirs: Optional[Sequence[Path | str]] = None,
    filename: Optional[str] = None,
    expected_sha256: Optional[str] = None,
) -> Path:
    local_dir = local_weight_dir(*Path(local_subdir).parts)
    target = local_dir / (filename or Path(url).name)

    if _is_valid_file(target, expected_sha256):
        return target

    for cache_dir in _unique_paths(cache_dirs or []):
        candidate = cache_dir / target.name
        if _is_valid_file(candidate, expected_sha256):
            return _copy_file(candidate, target)

    _download_to_file(url, target)
    if not _is_valid_file(target, expected_sha256):
        raise RuntimeError(f"Downloaded file has invalid checksum: {target}")
    return target


def resolve_hf_file(
    repo_id: str,
    filename: str,
    local_subdir: str | Path,
    *,
    repo_type: str = "model",
    revision: Optional[str] = None,
    alt_filenames: Optional[Sequence[str]] = None,
    legacy_local_files: Optional[Sequence[Path | str]] = None,
) -> Path:
    local_dir = local_weight_dir(*Path(local_subdir).parts)
    filenames = list(dict.fromkeys([*(alt_filenames or []), filename]))

    for name in filenames:
        local_file = local_dir / Path(name).name
        if local_file.is_file():
            return local_file

    for legacy_file in _unique_paths(legacy_local_files or []):
        if legacy_file.is_file():
            return _copy_file(legacy_file, local_dir / legacy_file.name)

    for cache_dir in hf_cache_dirs():
        for name in filenames:
            try:
                cached_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=name,
                    repo_type=repo_type,
                    revision=revision,
                    cache_dir=str(cache_dir),
                    local_files_only=True,
                )
            except Exception:
                continue
            return _copy_file(Path(cached_file), local_dir / Path(cached_file).name)

    last_error: Optional[Exception] = None
    for name in filenames:
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=name,
                repo_type=repo_type,
                revision=revision,
                local_dir=str(local_dir),
            )
            downloaded_path = Path(downloaded)
            target = local_dir / downloaded_path.name
            return _copy_file(downloaded_path, target)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Unable to resolve Hugging Face file {repo_id}/{filename}") from last_error


def resolve_hf_snapshot(
    repo_id: str,
    local_subdir: str | Path,
    *,
    repo_type: str = "model",
    revision: Optional[str] = None,
    allow_patterns: Optional[Sequence[str] | str] = None,
    required_paths: Optional[Sequence[str]] = None,
    required_any_paths: Optional[Sequence[str]] = None,
) -> Path:
    local_dir = local_weight_dir(*Path(local_subdir).parts)
    if _has_required_files(local_dir, required_paths, required_any_paths):
        return local_dir
    if not required_paths and not required_any_paths and any(local_dir.iterdir()):
        return local_dir

    for cache_dir in hf_cache_dirs():
        try:
            cached_dir = snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                cache_dir=str(cache_dir),
                allow_patterns=allow_patterns,
                local_files_only=True,
            )
        except Exception:
            continue
        _copy_tree(Path(cached_dir), local_dir)
        if _has_required_files(local_dir, required_paths, required_any_paths):
            return local_dir

    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
    )
    if not _has_required_files(local_dir, required_paths, required_any_paths):
        raise RuntimeError(f"Snapshot for {repo_id} is missing required files in {local_dir}")
    return local_dir


def resolve_transformers_source(model_name_or_path: str | Path) -> Path | str:
    path = Path(model_name_or_path)
    if path.exists():
        return path
    return resolve_hf_snapshot(
        str(model_name_or_path),
        Path("transformers") / sanitize_name(str(model_name_or_path)),
        required_paths=("config.json",),
        required_any_paths=(
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        ),
    )


def resolve_open_clip_source(model_id: str) -> Path:
    return resolve_hf_snapshot(
        model_id,
        Path("open_clip") / sanitize_name(model_id),
        required_paths=("open_clip_config.json",),
        required_any_paths=("open_clip_model.safetensors", "open_clip_pytorch_model.bin"),
    )


def resolve_timm_pretrained_overlay(model_name: str) -> Optional[dict[str, object]]:
    from timm.models._registry import get_pretrained_cfg

    pretrained_cfg = get_pretrained_cfg(model_name)
    if pretrained_cfg is None:
        return None

    cfg = pretrained_cfg.to_dict() if hasattr(pretrained_cfg, "to_dict") else dict(pretrained_cfg)

    if cfg.get("file"):
        file_path = Path(cfg["file"])
        if file_path.exists():
            if file_path.is_dir():
                state_dict = _load_timm_state_dict(file_path)
                if state_dict is not None:
                    return {"state_dict": state_dict}
                return {"source": "local-dir", "file": str(file_path)}
            if file_path.suffix == ".safetensors":
                return {"state_dict": _load_timm_state_dict(file_path)}
            return {"file": str(file_path)}

    if cfg.get("url"):
        local_file = resolve_url_weight(
            cfg["url"],
            Path("timm") / "checkpoints",
            cache_dirs=timm_url_cache_dirs(),
        )
        if local_file.suffix == ".safetensors":
            return {"state_dict": _load_timm_state_dict(local_file)}
        return {"file": str(local_file)}

    if cfg.get("hf_hub_id"):
        local_dir = resolve_hf_snapshot(
            cfg["hf_hub_id"],
            Path("timm") / "snapshots" / sanitize_name(cfg["hf_hub_id"]),
            required_any_paths=(
                "model.safetensors",
                "pytorch_model.bin",
                "open_clip_model.safetensors",
                "open_clip_pytorch_model.bin",
            ),
        )
        state_dict = _load_timm_state_dict(local_dir)
        if state_dict is not None:
            return {"state_dict": state_dict}
        return {"source": "local-dir", "file": str(local_dir)}

    return None
