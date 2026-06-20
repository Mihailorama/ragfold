import pytest

from ragfold.vectorstores import (
    ChromaVectorStore,
    FaissVectorStore,
    InMemoryVectorStore,
    PGVectorStore,
    QdrantVectorStore,
    VectorRecord,
)


def test_in_memory_vector_store_returns_cosine_nearest_records():
    store = InMemoryVectorStore()
    store.add(
        [
            VectorRecord(id="a", vector=[1.0, 0.0], text="alpha"),
            VectorRecord(id="b", vector=[0.0, 1.0], text="beta"),
        ]
    )

    matches = store.query([0.9, 0.1], top_k=2)

    assert [match.id for match in matches] == ["a", "b"]
    assert matches[0].score > matches[1].score


def test_in_memory_vector_store_rejects_dimension_mismatch():
    store = InMemoryVectorStore()
    store.add([VectorRecord(id="a", vector=[1.0, 0.0], text="alpha")])

    with pytest.raises(ValueError, match="dimension"):
        store.query([1.0, 0.0, 0.0], top_k=1)


def test_optional_vector_stores_are_gated_and_constructible():
    stores = [FaissVectorStore(), QdrantVectorStore(), ChromaVectorStore(), PGVectorStore()]

    for store in stores:
        assert isinstance(store.is_available(), bool)
        if not store.is_available():
            with pytest.raises(NotImplementedError, match="Install"):
                store.query([1.0], top_k=1)
