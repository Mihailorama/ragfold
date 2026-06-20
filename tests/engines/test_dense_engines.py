import pytest

from ragfold.engines.dense import (
    CohereEmbedEngine,
    OpenAIEmbedEngine,
    SentenceTransformersEngine,
    VoyageEmbedEngine,
)


class ToyEmbedder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append([
                1.0 if "invoice" in lower else 0.0,
                1.0 if "contract" in lower else 0.0,
            ])
        return vectors


@pytest.mark.asyncio
async def test_sentence_transformers_engine_uses_injected_embedder():
    engine = SentenceTransformersEngine(model=ToyEmbedder())
    corpus = [
        {"id": "invoice", "text": "Invoice total and payment terms"},
        {"id": "contract", "text": "Contract indemnity clause"},
    ]

    result = await engine.retrieve(corpus, "invoice amount", top_k=1)

    assert engine.is_available() is True
    assert result.passages[0].document_id == "invoice"
    assert engine.capabilities.retrieval_method == "dense"


def test_saas_dense_engines_are_unavailable_without_injected_clients_or_keys(monkeypatch):
    for env_name in ["OPENAI_API_KEY", "COHERE_API_KEY", "VOYAGE_API_KEY"]:
        monkeypatch.delenv(env_name, raising=False)

    engines = [OpenAIEmbedEngine(), CohereEmbedEngine(), VoyageEmbedEngine()]

    for engine in engines:
        assert engine.is_available() is False
        with pytest.raises(NotImplementedError, match="unavailable"):
            engine.ensure_available()
