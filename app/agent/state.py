from typing import TypedDict, Annotated, Optional
from langchain_core.documents import Document
import operator

class AgentState(TypedDict):
    # --- Inputs ---
    question: str
    # The original question is preserved for reference even after rewriting
    original_question: str

    # --- Retrieval ---
    documents: list[Document]
    # Document objects carry .page_content (the text) and .metadata (source, s3_key, etc.)
    # Using LangChain's Document type here because it integrates with LangChain's
    # prompt templates directly — you can pass a list of Documents into a chain
    # and it knows how to format them automatically.

    # --- Grading ---
    all_docs_relevant: bool
    # True if ALL retrieved documents passed the relevance check.
    # This drives the conditional edge decision.

    # --- Loop Control ---
    retrieval_attempts: int
    # Tracks how many times we've tried to retrieve.
    # Critical for the safety limit — without this, a bad query loops forever.
    max_attempts: int
    # Set at graph invocation time. Default: 3.
    # Why 3? Each attempt makes 2 LLM calls (grade + rewrite) plus a Pinecone query.
    # At 3 attempts you're already at ~$0.001 per query — enough to catch bad queries
    # without runaway costs.

    # --- Generation ---
    answer: Optional[str]
    sources: list[str]
    # List of source strings (URLs, document names) that were actually used
    # in generating the answer. This is what powers the "Sources" sidebar
    # in your Iteration 3 frontend.

    # --- Metadata (useful for debugging and Iteration 5 tracing) ---
    was_query_rewritten: bool
    rewritten_questions: list[str]
    # Full history of rewritten queries — useful for debugging and for showing
    # users "we also searched for: ..." in the UI