from ragfold.engines.base import (
    DocumentChunk,
    EngineCapabilities,
    RagAnswer,
    RetrievedPassage,
    RetrievalResult,
    normalize_corpus,
)


def test_retrieval_result_serializes_public_fields():
    result = RetrievalResult(
        engine_name="text-rag",
        query="capital of France",
        passages=[
            RetrievedPassage(
                document_id="doc-1",
                text="Paris is the capital of France.",
                score=0.95,
                rank=1,
                metadata={"page": 1},
            )
        ],
        answer=RagAnswer(answer="Paris", citations=["doc-1"], confidence=0.9),
        processing_time_ms=12,
        token_cost=0.001,
    )

    as_dict = result.to_dict()

    assert as_dict["engine_name"] == "text-rag"
    assert as_dict["passages"][0]["document_id"] == "doc-1"
    assert as_dict["answer"]["answer"] == "Paris"
    assert as_dict["token_cost"] == 0.001


def test_engine_capabilities_describe_rag_surface():
    caps = EngineCapabilities(
        modality="visual",
        retrieval_method="late-interaction",
        ocr_free=True,
        local=True,
        rerank=True,
    )

    assert caps.to_dict()["modality"] == "visual"
    assert caps.to_dict()["ocr_free"] is True
    assert caps.to_dict()["rerank"] is True


def test_normalize_corpus_accepts_strings_dicts_and_chunks():
    corpus = [
        "plain text",
        {"id": "doc-2", "text": "dict text", "metadata": {"source": "fixture"}},
        DocumentChunk(id="doc-3", text="chunk text"),
    ]

    chunks = normalize_corpus(corpus)

    assert [chunk.id for chunk in chunks] == ["doc-1", "doc-2", "doc-3"]
    assert chunks[1].metadata["source"] == "fixture"
