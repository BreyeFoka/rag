from types import SimpleNamespace

import pytest

from src.sparse_retrieval import BM25Retriever

CHUNKS = [
    "the solar panel charges the battery during the day",
    "the battery powers the pump at night",
    "the pump moves water from the pond to the filter",
    "the filter removes algae from the water",
]


class FakeDB:
    """Stands in for DatabaseManager: one document holding the chunks above."""

    def get_all_documents(self):
        return [SimpleNamespace(id=1)]

    def get_chunks_by_document(self, document_id):
        return [
            SimpleNamespace(id=i, content=text, document_id=document_id, chunk_index=i)
            for i, text in enumerate(CHUNKS)
        ]


@pytest.fixture
def retriever(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the index is written under ./data
    r = BM25Retriever(FakeDB())
    r.build_index()
    return r


def test_ranks_the_chunk_with_the_rare_term_first(retriever):
    results = retriever.search("algae", top_k=3)
    assert results[0]["content"] == CHUNKS[3]
    assert len(results) == 1


def test_idf_is_positive_even_for_very_common_terms(retriever):
    # "the" is in every chunk; the classic BM25 IDF would make it negative.
    assert retriever.get_idf("the") > 0
    assert retriever.get_idf("algae") > retriever.get_idf("the")


def test_matching_a_common_word_never_lowers_a_score(retriever):
    terms = retriever.tokenize("pump")
    with_common = retriever.tokenize("the pump")
    for i in range(len(CHUNKS)):
        assert retriever.score_document(with_common, i) >= retriever.score_document(terms, i)


def test_index_survives_a_reload(retriever, tmp_path):
    reloaded = BM25Retriever(FakeDB())
    assert len(reloaded.documents) == len(CHUNKS)
    assert reloaded.search("battery")[0]["id"] in {"0", "1"}


def test_empty_query_returns_nothing(retriever):
    assert retriever.search("   ") == []
