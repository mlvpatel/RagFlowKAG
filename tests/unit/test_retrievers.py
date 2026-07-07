"""Unit tests for hybrid retrieval and reranking (no database, no torch)."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.retrieval.retrievers import HybridRetriever, ReRankingRetriever


class _FakeBase:
    def __init__(self, docs):
        self.docs = docs

    def invoke(self, query):
        return self.docs


class _FakeEncoder:
    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs):
        return self.scores


class _FakeEmbeddings:
    def embed_query(self, query):
        return [0.0] * 768


def test_reranking_orders_by_score_and_truncates_to_top_n():
    docs = [
        Document(page_content="a"),
        Document(page_content="b"),
        Document(page_content="c"),
    ]
    retriever = ReRankingRetriever(
        base_retriever=_FakeBase(docs),
        top_n=2,
        cross_encoder=_FakeEncoder([0.1, 0.9, 0.5]),
    )
    out = retriever.invoke("q")
    assert [d.page_content for d in out] == ["b", "c"]


def test_reranking_empty_result_passes_through():
    retriever = ReRankingRetriever(
        base_retriever=_FakeBase([]), top_n=3, cross_encoder=_FakeEncoder([])
    )
    assert retriever.invoke("q") == []


def test_hybrid_retriever_maps_rows_to_documents_and_uses_rrf_sql():
    rows = [
        ("id1", "doc one", {"file_id": 1}, 0.5),
        ("id2", "doc two", {"file_id": 2}, 0.3),
    ]
    conn_cm = MagicMock()
    conn = conn_cm.__enter__.return_value
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = rows

    with patch("src.retrieval.retrievers.psycopg.connect", return_value=conn_cm):
        retriever = HybridRetriever(embeddings=_FakeEmbeddings())
        out = retriever.invoke("hello")

    assert [d.page_content for d in out] == ["doc one", "doc two"]
    assert out[0].metadata["file_id"] == 1
    assert out[0].metadata["score"] == 0.5
    executed_sql = cur.execute.call_args[0][0]
    assert "FULL OUTER JOIN" in executed_sql
    assert "plainto_tsquery" in executed_sql
