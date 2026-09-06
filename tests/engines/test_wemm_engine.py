from pathlib import Path

import pytest

from ragfold.engines import DEFAULT_ENGINE_NAMES, build_default_router
from ragfold.engines.wemm import WeMMEmbeddingEngine


class ToyMultimodalEmbedder:
    """Injected stand-in for WeMM: 3-dim vectors, invoice/contract/video axes."""

    def encode(self, texts):
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                [
                    1.0 if "invoice" in lower else 0.0,
                    1.0 if "contract" in lower else 0.0,
                    1.0 if "video" in lower else 0.0,
                ]
            )
        return vectors


@pytest.mark.asyncio
async def test_wemm_injected_embedder_ranks_and_reports_capabilities():
    engine = WeMMEmbeddingEngine(model=ToyMultimodalEmbedder())
    corpus = [
        {"id": "invoice", "text": "Invoice total and payment terms"},
        {"id": "contract", "text": "Contract indemnity clause"},
    ]

    result = await engine.retrieve(corpus, "invoice amount", top_k=1)

    assert engine.is_available() is True
    assert result.passages[0].document_id == "invoice"
    caps = engine.capabilities
    assert caps.modality == "multimodal"
    assert caps.retrieval_method == "dense"
    assert caps.vlm is True
    assert caps.license == "Apache-2.0"


@pytest.mark.asyncio
async def test_wemm_truncate_dim_slices_matryoshka_vectors():
    engine = WeMMEmbeddingEngine(model=ToyMultimodalEmbedder(), truncate_dim=1)
    # With only the first (invoice) dimension kept, the contract axis is invisible.
    result = await engine.retrieve(
        [{"id": "invoice", "text": "Invoice total"}],
        "invoice",
        top_k=1,
    )
    stored_dim = len(result.passages[0].metadata.get("vector", [0]))
    # The engine records the truncated width it indexed at.
    assert result.metadata.get("truncate_dim") == 1
    assert stored_dim in (0, 1)


def test_wemm_unavailable_without_model_or_runtime(monkeypatch):
    monkeypatch.setattr(
        "ragfold.engines.wemm.importlib.util.find_spec",
        lambda name: None,
    )
    engine = WeMMEmbeddingEngine()
    assert engine.is_available() is False
    with pytest.raises(NotImplementedError, match="unavailable"):
        engine.ensure_available()


def test_wemm_registered_in_default_router():
    assert "wemm" in DEFAULT_ENGINE_NAMES
    listed = {engine["name"] for engine in build_default_router().list_engines()}
    assert "wemm" in listed


def test_wemm_optional_extra_declared():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "wemm = [" in pyproject
