"""RagFlowKAG Streamlit application entry point."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st  # noqa: E402

from frontend.chat_interface import display_chat_interface  # noqa: E402
from frontend.sidebar import display_sidebar  # noqa: E402


def main() -> None:
    st.set_page_config(page_title="RagFlowKAG", layout="wide")
    st.title("RagFlowKAG")
    st.caption(
        "Knowledge augmented RAG, 2025. A knowledge graph built from your "
        "documents, linked to the answer."
    )
    display_sidebar()
    display_chat_interface()


if __name__ == "__main__":
    main()
