"""Fetch the external source files that the analysis cannot derive itself.

The rest of this project is offline: `src/analysis.py` and the figure modules
never touch the network, and every input they need must already be present in
`input/`. This module is the one deliberate exception, kept separate so that
no analysis or figure recipe can acquire a network dependency by accident.
Run it once via `just fetch-inputs`; it is not part of `just all`.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import tqdm

from src import settings

CHUNK_BYTES = 1 << 20


def download(url: str, destination: Path, refresh_download: bool = False) -> Path:
    """Download `url` to `destination`, skipping the transfer if it already exists.

    The file is streamed to a uniquely named temporary sibling and installed
    with `os.replace` only after the transfer completes, so an interrupted run
    cannot leave a truncated file that later looks usable.

    Parameters
    ----------
    url
        Source address.
    destination
        Final path. Its parent directory is created if needed.
    refresh_download
        Re-download even when `destination` exists. Source downloads are never
        refreshed by the analysis `force_redo` flags, because repeating a large
        transfer to recompute a derived product is rarely what the caller
        wants.

    Returns
    -------
    pathlib.Path
        The finalized `destination`.
    """
    if destination.exists() and not refresh_download:
        print(f"{destination.name} already present; pass refresh_download=True to re-fetch")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with urllib.request.urlopen(url) as response:  # noqa: S310 - fixed https URL from settings
            total = int(response.headers.get("Content-Length", 0)) or None
            with (
                temporary.open("wb") as handle,
                tqdm.tqdm(total=total, unit="B", unit_scale=True, desc=destination.name) as progress,
            ):
                while chunk := response.read(CHUNK_BYTES):
                    handle.write(chunk)
                    progress.update(len(chunk))
        os.replace(temporary, destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def run(refresh_download: bool = False) -> None:
    """Fetch every external source file the project depends on."""
    download(settings.MASS_CHANGE_URL, settings.MASS_CHANGE_FILE, refresh_download=refresh_download)
    download(settings.DEM_URL, settings.DEM_FILE, refresh_download=refresh_download)


if __name__ == "__main__":
    run()
