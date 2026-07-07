"""Unit tests for the vector store utilities (no database, no network)."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

import src.embeddings.vectorstore_utils as vs


def test_text_splitter_chunks_a_long_document():
    doc = Document(page_content="word " * 3000, metadata={})
    chunks = vs._get_text_splitter().split_documents([doc])
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= vs.settings.chunk_size + 50


def test_index_document_tags_each_chunk_with_file_id_and_filename(monkeypatch):
    canned = [
        Document(page_content="a", metadata={}),
        Document(page_content="b", metadata={}),
    ]
    monkeypatch.setattr(vs, "load_and_split_document", lambda path: canned)
    store = MagicMock()
    monkeypatch.setattr(vs, "get_store", lambda: store)

    ok = vs.index_document("/tmp/report.txt", 7)

    assert ok is True
    store.add_documents.assert_called_once()
    passed = store.add_documents.call_args[0][0]
    assert all(d.metadata["file_id"] == 7 for d in passed)
    assert all(d.metadata["filename"] == "report.txt" for d in passed)


def test_index_document_returns_false_on_error(monkeypatch):
    def boom(_):
        raise RuntimeError("loader failed")

    monkeypatch.setattr(vs, "load_and_split_document", boom)
    assert vs.index_document("/tmp/report.txt", 7) is False


def test_delete_doc_runs_a_collection_scoped_sql_delete():
    conn_cm = MagicMock()
    conn = conn_cm.__enter__.return_value
    cur = conn.cursor.return_value.__enter__.return_value

    with patch(
        "src.embeddings.vectorstore_utils.psycopg.connect", return_value=conn_cm
    ):
        ok = vs.delete_doc(7)

    assert ok is True
    sql, params = cur.execute.call_args[0]
    assert "DELETE FROM langchain_pg_embedding" in sql
    assert params == (vs.COLLECTION_NAME, "7")
