"""Text chunking for ingestion pipeline."""


def chunk_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50,
    respect_sentences: bool = True,
) -> list[str]:
    """
    Split text into overlapping chunks by word count.

    Overlap ensures the end of one chunk and the start of the next share context,
    so embeddings stay accurate across boundaries. When respect_sentences is True,
    chunk boundaries prefer sentence ends to avoid cutting sentences in half.

    Args:
        text: Raw document text.
        chunk_size: Target words per chunk (before overlap).
        overlap: Number of words to overlap between consecutive chunks.
        respect_sentences: If True, try to end chunks at sentence boundaries.

    Returns:
        Ordered list of chunk strings.
    """
    # If the text is empty, return an empty list
    if not text or not text.strip():
        return []

    # If the overlap is greater than the chunk size, set the overlap to the chunk size - 1
    if overlap >= chunk_size:
        overlap = max(0, chunk_size - 1)

    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        window = words[start:end]
        chunk_str = " ".join(window)

        if respect_sentences and end < len(words):
            # Prefer to end at last sentence boundary in this window
            last_sentence_end = _last_sentence_end(chunk_str)
            if last_sentence_end is not None and last_sentence_end > chunk_size // 2:
                chunk_str = chunk_str[: last_sentence_end + 1].strip()
                # Adjust start so next chunk begins after this sentence (by word count)
                trimmed_words = len(chunk_str.split())
                end = start + trimmed_words

        chunks.append(chunk_str)
        # Next window: step back by overlap so we get overlap words shared
        next_start = end - overlap
        start = max(next_start, start + 1)  # always advance at least 1 word
        if start >= len(words):
            break

    return [c for c in chunks if c.strip()]


def _last_sentence_end(s: str) -> int | None:
    """Return index of the last sentence-ending punctuation (. ! ?) in s, or None."""
    last = -1
    for sep in (". ", "! ", "? "):
        idx = s.rfind(sep)
        if idx != -1:
            last = max(last, idx)
    return last if last != -1 else None
