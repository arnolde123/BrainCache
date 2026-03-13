"""Centralized prompt templates for the agent workflow."""

SYSTEM_PROMPT = (
    "You are BrainCache assistant. Answer using retrieved context when available."
)

GRADING_PROMPT = (
    "Given a user query and retrieved chunks, decide if context is sufficient to answer."
)

REWRITE_PROMPT = (
    "Rewrite the user query to improve retrieval recall while preserving intent."
)

GENERATION_PROMPT = (
    "Generate a concise and accurate answer grounded in retrieved context."
)
