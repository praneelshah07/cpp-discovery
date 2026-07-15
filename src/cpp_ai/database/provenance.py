"""File-level provenance helpers.

Reproducibility requires knowing *exactly* which bytes produced a dataset, so
every import records the SHA-256 of its source file. Hashing is streamed so
multi-hundred-megabyte database dumps do not need to fit in memory.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE = 1 << 20  # 1 MiB


def file_sha256(path: str | Path) -> str:
    """Return the hex SHA-256 digest of a file's contents.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    p = Path(path)
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()
