"""LLM extraction of knowledge triples and query entities for rag-graph-2024.

Both use a small local model by default, so indexing and querying stay keyless.
Output is constrained to compact JSON and parsed defensively, so a malformed
generation degrades to an empty result rather than crashing the pipeline.
"""

import json
import logging
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

_TRIPLES_SYSTEM = (
    "You extract factual knowledge triples from text. Return ONLY compact JSON: "
    '{"triples": [{"subject": "...", "predicate": "...", "object": "..."}]}. '
    "Use a short noun phrase for subject and object and a short verb phrase for "
    "predicate. Capture only facts stated in the text, such as policies, "
    "amounts, definitions, and relationships. Return at most %d triples."
)

_ENTITIES_SYSTEM = (
    "You list the key entities in a question: the people, organizations, "
    "products, policies, or amounts it is about. Return ONLY compact JSON: "
    '{"entities": ["...", "..."]}.'
)


def _parse(text: str, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(text[text.index("{") : text.rindex("}") + 1])
    except Exception:
        return default


def extract_triples(llm, text: str, max_triples: int = 12) -> List[Dict[str, str]]:
    """Extract up to max_triples factual triples from a chunk of text."""
    raw = llm.invoke(
        [
            SystemMessage(content=_TRIPLES_SYSTEM % max_triples),
            HumanMessage(content=text),
        ]
    ).content
    triples = _parse(raw, {"triples": []}).get("triples", [])
    clean = []
    for triple in triples:
        if (
            isinstance(triple, dict)
            and triple.get("subject")
            and triple.get("predicate")
            and triple.get("object")
        ):
            clean.append(
                {
                    "subject": str(triple["subject"]),
                    "predicate": str(triple["predicate"]),
                    "object": str(triple["object"]),
                }
            )
    return clean[:max_triples]


def extract_entities(llm, question: str) -> List[str]:
    """Extract the key entities a question is about, for graph linking."""
    raw = llm.invoke(
        [SystemMessage(content=_ENTITIES_SYSTEM), HumanMessage(content=question)]
    ).content
    entities = _parse(raw, {"entities": []}).get("entities", [])
    return [str(entity) for entity in entities if entity][:8]
