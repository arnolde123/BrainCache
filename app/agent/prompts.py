"""Prompt templates used by LangGraph agent nodes."""

GRADE_DOCUMENTS_PROMPT = """You are a grader assessing whether a retrieved document
is relevant to a user question.

Retrieved document:
{document}

User question: {question}

Give a binary score of 'yes' or 'no' to indicate whether the document is relevant
to the question.

Rules:
- 'yes' if the document contains information that would help answer the question,
  even partially.
- 'no' if the document is on a completely different topic.
- Be permissive - if there's any reasonable connection, score 'yes'.
- Do not consider whether the document fully answers the question, only whether
  it is relevant.

Respond with ONLY the JSON: {{"score": "yes"}} or {{"score": "no"}}
No explanation, no preamble."""

REWRITE_QUERY_PROMPT = """You are a query rewriter. Your job is to improve a search
query that failed to retrieve useful results from a vector database.

Original user question: {original_question}
Current (failed) query: {current_question}
Attempt number: {attempt}

The query failed because the retrieved documents were not relevant.
Rewrite the query to:
1. Use different vocabulary (synonyms, related terms)
2. Be more specific if the original was vague
3. Break a complex question into its most searchable core concept
4. Consider what words an author would use when writing about this topic

Previous rewrites (don't repeat these): {previous_rewrites}

Respond with ONLY the rewritten query as a plain string. No explanation, no quotes."""

GENERATE_PROMPT = """You are an assistant answering questions based strictly on the
provided context documents.

Context documents:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the context provided. Do not use outside knowledge.
- If the context does not contain enough information, say so explicitly.
- Be concise and direct.
- After your answer, list the sources you used as "Sources: [source1, source2]"
- If you used information from a specific document, you must cite it.

Answer:"""
