import pytest


def test_recursive_chunker_splits_text_without_optional_dependencies():
    from ragfold.preprocessing import RecursiveChunker

    chunker = RecursiveChunker(chunk_size=5, chunk_overlap=1)
    chunks = chunker.chunk("alpha beta gamma delta epsilon zeta", document_id="doc")

    assert [chunk.document_id for chunk in chunks] == ["doc", "doc"]
    assert chunks[0].text == "alpha beta gamma delta epsilon"
    assert chunks[1].text == "epsilon zeta"


def test_adaptive_chunker_is_gated_when_dependency_missing():
    from ragfold.preprocessing import AdaptiveChunker

    chunker = AdaptiveChunker()

    if not chunker.is_available():
        with pytest.raises(NotImplementedError, match="adaptive-chunking"):
            chunker.chunk("text", document_id="doc")


def test_noop_and_headroom_compressors_share_interface():
    from ragfold.compression import HeadroomCompressor, NoopCompressor
    from ragfold.engines.base import RetrievedPassage

    passages = [
        RetrievedPassage(document_id="doc", text="Important retrieved context", score=1.0, rank=1)
    ]

    compressed = NoopCompressor().compress(passages, query="context")

    assert compressed[0].text == "Important retrieved context"

    headroom = HeadroomCompressor()
    if not headroom.is_available():
        with pytest.raises(NotImplementedError, match="headroom"):
            headroom.compress(passages, query="context")


def test_cocoindex_indexer_is_gated_when_dependency_missing():
    from ragfold.indexing import CocoIndexIndexer

    indexer = CocoIndexIndexer()

    if not indexer.is_available():
        with pytest.raises(NotImplementedError, match="cocoindex"):
            indexer.index([{"id": "doc", "text": "hello"}])
