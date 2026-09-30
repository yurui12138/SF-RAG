"""Prompts used by SF-RAG indexing, retrieval extensions and generation."""


def summarize(title, path, content, previous_summary):
    return (
        "Summarize this academic segment in 20-40 words. Preserve technical evidence, "
        "its position in the paper, and continuity with the preceding segment. "
        "Do not add unsupported information. Return only the summary.\n"
        f"Paper title: {title}\nSection path: {path}\n"
        f"Previous segment summary: {previous_summary}\nCurrent segment: {content}"
    )


def generate_answer(question, context):
    return (
        "Answer the academic question using only the retrieved evidence. Be concise "
        "and accurate; reconcile multiple papers when applicable. Answer in the "
        "question's language. If evidence is insufficient, say so.\n"
        f"Question: {question}\nRetrieved evidence:\n{context}\nAnswer:"
    )


def decompose_question(question):
    return (
        "For optional multi-hop retrieval, decompose this question into at most "
        "three ordered, focused subqueries. Use {previous_entity} where a later "
        "step depends on an entity from an earlier step. For a single-hop question "
        "return a one-element array. Return ONLY a JSON array of strings.\n"
        f"Question: {question}"
    )


def transform_question(question):
    return (
        "Rewrite this multi-document question as one canonical question applicable "
        "independently to each individual paper. Return only the transformed question.\n"
        f"Question: {question}"
    )
