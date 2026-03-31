from src.knowledge_base import LegalKnowledgeBase


def test_retrieve_returns_relevant_documents():
    kb = LegalKnowledgeBase()

    results = kb.retrieve(
        query="Either party may terminate immediately without notice.",
        clause_type="termination",
        jurisdiction="california",
        k=2,
    )

    assert results
    assert any(doc.metadata.get("type") in ("termination", "general") for doc in results)


def test_retrieve_prefers_matching_jurisdiction():
    kb = LegalKnowledgeBase()

    results = kb.retrieve(
        query="California service contract termination without notice.",
        clause_type="termination",
        jurisdiction="california",
        k=3,
    )

    assert results
    assert any(doc.metadata.get("jurisdiction") == "california" for doc in results)
