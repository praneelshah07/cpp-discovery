"""Real ESM-2 integration test.

Skipped automatically unless fair-esm + torch are installed. Uses the tiny
8M-parameter checkpoint (fast, small download) rather than the 650M default.
Run explicitly with: pytest -m slow
"""

from __future__ import annotations

import numpy as np
import pytest

esm = pytest.importorskip("esm")  # noqa: F841  -- gates the whole module
pytest.importorskip("torch")

from cpp_ai.embeddings import ESM2Embedder  # noqa: E402

pytestmark = pytest.mark.slow


def test_esm2_8m_embeds_with_expected_shape() -> None:
    embedder = ESM2Embedder("esm2_t6_8M")
    assert embedder.dim == 320
    vecs = embedder.embed_many(["KWKLFKKI", "LLIILRRRIRKQAHAHSK"])
    assert vecs.shape == (2, 320)
    assert vecs.dtype == np.float32
    assert np.isfinite(vecs).all()


def test_esm2_is_deterministic() -> None:
    embedder = ESM2Embedder("esm2_t6_8M")
    a = embedder.embed("KWKLFKKI")
    b = embedder.embed("KWKLFKKI")
    assert np.allclose(a, b, atol=1e-5)


def test_esm2_distinguishes_sequences() -> None:
    embedder = ESM2Embedder("esm2_t6_8M")
    a = embedder.embed("KWKLFKKI")
    b = embedder.embed("DDDDEEEE")
    assert not np.allclose(a, b)
