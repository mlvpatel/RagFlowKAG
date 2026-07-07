"""Knowledge graph store for RagFlowKAG.

The graph is a single edge table of subject, predicate, object triples on the
same Postgres instance as pgvector, so no extra service is needed. Nodes are
implicit: any string that appears as a subject or an object. Retrieval matches
the question's entities against subjects and objects and pulls their one hop
neighbourhood, which is the structured half of knowledge augmented generation.

The connection pool is shared with src.api.db_utils and imported lazily inside
each function, so importing this module opens no database connection.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _pool():
    from src.api.db_utils import _get_pool

    return _get_pool()


def create_kg_edges() -> None:
    """Create the kg_edges table and its case insensitive lookup indexes."""
    with _pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS kg_edges (
                    id         BIGSERIAL PRIMARY KEY,
                    subject    TEXT NOT NULL,
                    predicate  TEXT NOT NULL,
                    object     TEXT NOT NULL,
                    file_id    BIGINT,
                    filename   TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_kg_subject ON kg_edges (lower(subject))"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_kg_object ON kg_edges (lower(object))"
            )


def insert_edges(edges: List[Dict[str, str]], file_id: int, filename: str) -> int:
    """Insert triples for one document. Returns the number stored."""
    rows = [
        (
            str(e["subject"]).strip(),
            str(e["predicate"]).strip(),
            str(e["object"]).strip(),
            file_id,
            filename,
        )
        for e in edges
        if e.get("subject") and e.get("predicate") and e.get("object")
    ]
    if not rows:
        return 0
    with _pool().connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO kg_edges (subject, predicate, object, file_id, filename) "
                "VALUES (%s, %s, %s, %s, %s)",
                rows,
            )
    return len(rows)


def delete_edges(file_id: int) -> bool:
    """Delete all triples for a document id."""
    try:
        with _pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM kg_edges WHERE file_id = %s", (file_id,))
        return True
    except Exception as exc:
        logger.error("Failed to delete edges for file_id %s: %s", file_id, exc)
        return False


def match_neighbors(entities: List[str], limit: int = 20) -> List[Dict[str, Any]]:
    """Return triples whose subject or object matches any of the entities.

    Matching is case insensitive and substring based, so an entity like "Acme"
    links to "Acme Corporation". Results are distinct and capped by limit. All
    values are bound as parameters, never interpolated into the SQL.
    """
    terms = [e.strip() for e in entities if e and e.strip()]
    if not terms:
        return []
    clauses = []
    params: List[Any] = []
    for term in terms:
        clauses.append("(subject ILIKE %s OR object ILIKE %s)")
        like = f"%{term}%"
        params.extend([like, like])
    params.append(limit)
    sql = (
        "SELECT DISTINCT subject, predicate, object, filename FROM kg_edges "
        f"WHERE {' OR '.join(clauses)} LIMIT %s"
    )
    with _pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    return [dict(row) for row in rows]


def count_edges() -> int:
    """Return the total number of triples in the graph."""
    with _pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM kg_edges")
            row = cur.fetchone()
    return row["n"] if isinstance(row, dict) else row[0]
