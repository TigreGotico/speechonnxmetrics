"""Resolve a model reference to a local ``.onnx`` file path.

Resolution rules used by the metric adapters:

1. an explicit local ``.onnx`` path (used as-is);
2. an ``http(s)://`` URL (downloaded once and cached);
3. a ``(hf_repo, filename)`` pair -> HuggingFace download pinned by revision.

Downloads are cached under ``$XDG_DATA_HOME/speechmetrics``
(``~/.local/share/speechmetrics``).
"""
from __future__ import annotations

import hashlib
import os
from typing import Optional


def xdg_data_home() -> str:
    return os.environ.get("XDG_DATA_HOME") or os.path.join(
        os.path.expanduser("~"), ".local", "share"
    )


def get_cache_dir(cache_dir: Optional[str] = None) -> str:
    path = cache_dir or os.path.join(xdg_data_home(), "speechmetrics")
    os.makedirs(path, exist_ok=True)
    return path


def is_onnx_path(name: str) -> bool:
    return name.lower().endswith(".onnx")


def is_url(name: str) -> bool:
    return name.startswith("http://") or name.startswith("https://")


def download_url(url: str, cache_dir: Optional[str] = None) -> str:
    """Download an ONNX file from a URL into the cache (skipped if already present)."""
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    folder = os.path.join(get_cache_dir(cache_dir), "url", digest)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, os.path.basename(url.split("?")[0]) or "model.onnx")
    if not os.path.isfile(dest):
        import urllib.request

        with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
            f.write(resp.read())
    return dest


def hf_download(
    repo: str,
    filename: str,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """Download ``filename`` from a HuggingFace repo, cached and pinned by revision."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo,
        filename=filename,
        revision=revision,
        cache_dir=os.path.join(get_cache_dir(cache_dir), "hf"),
    )


def resolve(
    ref: str,
    *,
    hf_repo: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """Resolve ``ref`` (a local path, a URL, or a filename within ``hf_repo``).

    When ``ref`` is a plain filename and ``hf_repo`` is given, the file is fetched
    from HuggingFace. A local ``.onnx`` path or an ``http(s)://`` URL is used directly.
    """
    if is_url(ref):
        return download_url(ref, cache_dir)
    if is_onnx_path(ref) and os.path.isfile(ref):
        return ref
    if hf_repo:
        return hf_download(hf_repo, ref, revision, cache_dir)
    if os.path.isfile(ref):
        return ref
    raise FileNotFoundError(
        f"cannot resolve model reference {ref!r} (no local file, URL, or hf_repo)"
    )
