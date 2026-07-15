"""Chat interface for the rag-graph-2024 Streamlit app.

Shows the answer plus an expandable trace of the two retrieval channels: the
dense vector chunks and the knowledge graph facts linked from the question's
entities, so the structured reasoning is visible rather than hidden.
"""

import streamlit as st

from frontend import api_utils

_STEP_LABEL = {
    "vector_retrieval": "Vector retrieval",
    "entity_linking": "Entity linking",
    "graph_expansion": "Graph expansion",
    "grounded_answer": "Grounded answer",
}


def _render_trace(steps, entities, facts) -> None:
    if not steps and not facts:
        return
    header = f"Knowledge graph: {len(facts)} fact(s) linked"
    if entities:
        header += f" from {', '.join(entities)}"
    with st.expander(header, expanded=False):
        path = ", ".join(
            _STEP_LABEL.get(step.get("step"), step.get("step", "")) for step in steps
        )
        if path:
            st.caption(path)
        if facts:
            st.markdown("**Facts used**")
            for fact in facts:
                st.markdown(f"- {fact}")
        else:
            st.caption("No graph facts linked; answered from the vector channel.")


def display_chat_interface() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("steps") or message.get("facts"):
                _render_trace(
                    message.get("steps", []),
                    message.get("entities", []),
                    message.get("facts", []),
                )

    prompt = st.chat_input("Ask a question about your documents")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        model = st.session_state.get("model", "qwen2.5:7b-instruct")
        with st.spinner("Retrieving and reasoning over the graph..."):
            try:
                result = api_utils.chat(prompt, st.session_state.session_id, model)
            except Exception as exc:
                st.error(f"Request failed: {exc}")
                return
        st.session_state.session_id = result.get("session_id")
        answer = result.get("answer", "")
        st.markdown(answer)
        _render_trace(
            result.get("steps", []),
            result.get("entities", []),
            result.get("facts", []),
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "steps": result.get("steps", []),
            "entities": result.get("entities", []),
            "facts": result.get("facts", []),
        }
    )
