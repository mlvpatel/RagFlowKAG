"""Integration test: real pgvector round trip (index then delete by file_id).

Uses a deterministic fake embedder so the store round trip runs without any
embedding API key. Only Postgres with pgvector is required.
"""

import psycopg
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_postgres import PGVector

import src.embeddings.vectorstore_utils as vs
from src.core.config import settings

FILE_ID = 999001


def _count_chunks(file_id: int) -> int:
    with psycopg.connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*)
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = %s AND e.cmetadata->>'file_id' = %s
                """,
                (vs.COLLECTION_NAME, str(file_id)),
            )
            return cur.fetchone()[0]


def test_index_then_delete_round_trip(pg_available, monkeypatch):
    store = PGVector(
        embeddings=DeterministicFakeEmbedding(size=settings.embedding_dims),
        collection_name=vs.COLLECTION_NAME,
        connection=vs._sqlalchemy_url(),
        use_jsonb=True,
    )
    monkeypatch.setattr(vs, "_store", store)
    monkeypatch.setattr(
        vs,
        "load_and_split_document",
        lambda path: [
            Document(page_content="alpha beta gamma", metadata={}),
            Document(page_content="delta epsilon zeta", metadata={}),
        ],
    )

    # clean any leftover from a previous run
    vs.delete_doc(FILE_ID)
    assert _count_chunks(FILE_ID) == 0

    assert vs.index_document("dummy.txt", FILE_ID) is True
    assert _count_chunks(FILE_ID) == 2

    assert vs.delete_doc(FILE_ID) is True
    assert _count_chunks(FILE_ID) == 0
