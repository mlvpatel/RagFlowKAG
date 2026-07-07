"""Unit tests for the knowledge graph extraction parsing, in isolation.

No model or database is needed, so these run in CI without Ollama or Postgres.
"""

from src.kag.extract import _parse


def test_parse_extracts_json_object():
    parsed = _parse('noise {"entities": ["Acme", "Nimbus"]} tail', {})
    assert parsed == {"entities": ["Acme", "Nimbus"]}


def test_parse_falls_back_on_bad_json():
    default = {"triples": []}
    assert _parse("not json at all", default) == default
