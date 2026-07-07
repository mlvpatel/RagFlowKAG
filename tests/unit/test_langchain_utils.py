"""Unit tests for the RAG chain (fake chat models, no network)."""

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage

import src.core.langchain_utils as lu


class _FakeRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.last_query = None

    def invoke(self, query):
        self.last_query = query
        return self.docs


def test_answer_question_grounds_on_retrieved_docs_and_skips_reformulation(monkeypatch):
    fake_llm = FakeListChatModel(responses=["ANSWER"])
    retriever = _FakeRetriever([Document(page_content="ctx")])
    monkeypatch.setattr(lu, "_make_llm", lambda model: fake_llm)
    monkeypatch.setattr(lu, "get_final_retriever", lambda: retriever)

    out = lu.answer_question("gpt-4o-mini", "hello")

    assert out["answer"] == "ANSWER"
    assert retriever.last_query == "hello"  # first turn, no reformulation
    assert len(out["context"]) == 1


def test_answer_question_reformulates_when_history_is_present(monkeypatch):
    fake_llm = FakeListChatModel(responses=["standalone question", "ANSWER"])
    retriever = _FakeRetriever([Document(page_content="ctx")])
    monkeypatch.setattr(lu, "_make_llm", lambda model: fake_llm)
    monkeypatch.setattr(lu, "get_final_retriever", lambda: retriever)

    history = [
        {"role": "human", "content": "earlier question"},
        {"role": "ai", "content": "earlier answer"},
    ]
    out = lu.answer_question("gpt-4o-mini", "follow up", history)

    assert retriever.last_query == "standalone question"
    assert out["answer"] == "ANSWER"


def test_stream_answer_yields_text(monkeypatch):
    fake_llm = GenericFakeChatModel(messages=iter([AIMessage(content="hello world")]))
    retriever = _FakeRetriever([Document(page_content="ctx")])
    monkeypatch.setattr(lu, "_make_llm", lambda model: fake_llm)
    monkeypatch.setattr(lu, "get_final_retriever", lambda: retriever)

    tokens = list(lu.stream_answer("gpt-4o-mini", "hi"))

    assert tokens
    assert "".join(tokens).strip() == "hello world"


def test_make_llm_routes_local_models_to_ollama(monkeypatch):
    monkeypatch.setattr(lu.settings, "ollama_base_url", "http://localhost:11434")
    llm = lu._make_llm("llama3.2:3b")
    assert llm.__class__.__name__ == "ChatOllama"
