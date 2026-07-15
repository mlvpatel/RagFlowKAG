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


class _FakeRetriever:
    def __init__(self, docs=None):
        self.docs = docs or []
        self.last_query = None

    def invoke(self, query):
        self.last_query = query
        return self.docs


def test_run_kag_uses_the_standalone_query_everywhere(monkeypatch):
    """With history present, retrieval, entity linking, and the answer must all
    see the reformulated standalone question, never the raw follow-up."""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    import src.kag.engine as engine

    fake_llm = FakeListChatModel(
        responses=[
            "who runs Acme Corporation?",  # reformulation
            '{"entities": ["Acme"]}',  # entity extraction
            "Jane Doe runs it.",  # grounded answer
        ]
    )
    retriever = _FakeRetriever()
    monkeypatch.setattr(engine, "_make_llm", lambda *a, **k: fake_llm)
    monkeypatch.setattr(engine, "get_final_retriever", lambda: retriever)
    monkeypatch.setattr(engine.graph_store, "match_neighbors", lambda *a, **k: [])

    history = [{"role": "human", "content": "tell me about Acme Corporation"}]
    result = engine.run_kag("gpt-4o-mini", "and who runs it?", history)

    assert retriever.last_query == "who runs Acme Corporation?"
    assert result["entities"] == ["Acme"]
    assert result["answer"] == "Jane Doe runs it."
    assert any(s["step"] == "reformulate" for s in result["steps"])


def test_run_kag_skips_reformulation_without_history(monkeypatch):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    import src.kag.engine as engine

    fake_llm = FakeListChatModel(responses=['{"entities": []}', "answer text"])
    retriever = _FakeRetriever()
    monkeypatch.setattr(engine, "_make_llm", lambda *a, **k: fake_llm)
    monkeypatch.setattr(engine, "get_final_retriever", lambda: retriever)
    monkeypatch.setattr(engine.graph_store, "match_neighbors", lambda *a, **k: [])

    result = engine.run_kag("gpt-4o-mini", "what is Acme?", None)

    assert retriever.last_query == "what is Acme?"
    assert not any(s["step"] == "reformulate" for s in result["steps"])
    assert result["answer"] == "answer text"
