"""Knowledge augmented generation engine for rag-graph-2024.

Retrieval is two channels. The vector channel pulls the most similar chunks, as
in every earlier rung. The graph channel links the question's entities into the
knowledge graph and pulls their one hop neighbourhood of facts. The answer is
grounded in both, and the linked triples are returned as the trace, so the
structured reasoning is visible rather than hidden.

Keyless on Ollama. If the graph is empty, the engine degrades gracefully to the
vector channel alone, so it always answers.
"""

import logging
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from src.core.config import settings
from src.core.langchain_utils import _make_llm, get_final_retriever
from src.embeddings.vectorstore_utils import load_and_split_document
from src.kag import graph_store
from src.kag.extract import extract_entities, extract_triples

logger = logging.getLogger(__name__)

_QA_SYSTEM = (
    "You answer questions for rag-graph-2024 using the retrieved document context "
    "and the knowledge graph facts below. Use only what is given. If neither the "
    "context nor the facts contain the answer, say you do not have that "
    "information rather than inventing one.\n\n"
    "Document context:\n{context}\n\n"
    "Knowledge graph facts:\n{facts}"
)


def _facts_text(edges: List[Dict[str, Any]]) -> str:
    if not edges:
        return "None."
    return "\n".join(
        f"- {edge['subject']} {edge['predicate']} {edge['object']}" for edge in edges
    )


def run_kag(model: str, question: str, chat_history=None) -> dict:
    """Answer a question with vector plus knowledge graph retrieval."""
    llm = _make_llm(model, temperature=0)
    steps: List[Dict[str, Any]] = []

    # 1. Vector channel: the dense chunks, same as every earlier rung.
    docs = get_final_retriever().invoke(question)
    steps.append({"step": "vector_retrieval", "chunks": len(docs)})

    # 2. Graph channel: link the question's entities into the knowledge graph.
    entities = extract_entities(llm, question)
    steps.append({"step": "entity_linking", "entities": entities})
    edges = graph_store.match_neighbors(entities, limit=settings.kag_neighbor_limit)
    steps.append({"step": "graph_expansion", "facts": len(edges)})

    # 3. Ground the answer in both channels.
    context = "\n\n".join(doc.page_content for doc in docs) or "None."
    system = _QA_SYSTEM.format(context=context, facts=_facts_text(edges))
    answer = llm.invoke(
        [SystemMessage(content=system), HumanMessage(content=question)]
    ).content
    steps.append({"step": "grounded_answer"})

    sources = sorted(
        {
            doc.metadata.get("filename")
            for doc in docs
            if doc.metadata and doc.metadata.get("filename")
        }
        | {edge.get("filename") for edge in edges if edge.get("filename")}
    )
    facts = [f"{e['subject']} {e['predicate']} {e['object']}" for e in edges]
    return {
        "answer": answer,
        "sources": sources,
        "entities": entities,
        "facts": facts,
        "steps": steps,
    }


def build_graph(
    file_path: str,
    file_id: int,
    filename: str,
    model: str | None = None,
    max_chunks: int = 10,
) -> int:
    """Extract triples from a document's chunks and store them in the graph.

    Runs at index time. Best effort: any extraction failure is logged and
    skipped so it never blocks the vector indexing that already succeeded.
    """
    from src.core.config import settings

    model = model or settings.kag_extract_model
    llm = _make_llm(model, temperature=0)
    try:
        splits = load_and_split_document(file_path)
    except Exception as exc:
        logger.error("KAG could not load %s: %s", file_path, exc)
        return 0

    stored = 0
    for split in splits[:max_chunks]:
        try:
            triples = extract_triples(llm, split.page_content)
            stored += graph_store.insert_edges(triples, file_id, filename)
        except Exception as exc:
            logger.warning("KAG extraction skipped a chunk in %s: %s", filename, exc)
    logger.info("KAG stored %d triples for file_id=%s", stored, file_id)
    return stored
