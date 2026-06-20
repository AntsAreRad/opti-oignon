"""S185 audit fix RS-01 -- document the ChromaDB-at-rest gap (Option A).

The RAG vector store (ChromaDB, under data/chroma_v2/) keeps ingested chunk text
and embedding vectors in plaintext at rest -- ChromaDB has no native at-rest
encryption -- while the rest of the project is SQLCipher-everywhere. RS-01 is
design-scale (application-layer encryption of the vector store is a re-architecture
of the RAG ingest/retrieve paths plus per-corpus key management), so S185 takes the
documentation route: full-disk encryption (LUKS) is stated as a deployment
requirement for the RAG corpus, with an in-code note at the ChromaDB client, and
the encrypt-before-upsert work (Option B) is recorded as a future cycle.

These are source-content assertions that lock that documentation in place so the
known gap and its requirement cannot be silently dropped. They are not a runtime
test (there is no runtime change to assert).
"""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (_REPO_ROOT / rel).read_text(encoding="utf-8").lower()


def test_security_md_documents_chroma_at_rest_and_luks_requirement():
    text = _read("SECURITY.md")
    assert "rs-01" in text
    assert "chromadb" in text
    # The at-rest exposure is stated.
    assert "at rest" in text or "at-rest" in text
    assert "plaintext" in text
    # LUKS / full-disk encryption is stated as a requirement for the RAG corpus.
    assert "luks" in text
    assert "full-disk encryption" in text
    assert "requirement" in text


def test_rag_store_has_in_code_note():
    text = _read("opti_oignon/rag_store.py")
    assert "rs-01" in text
    assert "plaintext" in text
    # The note ties the gap to the deployment requirement / the planned cycle.
    assert "luks" in text or "full-disk" in text
    assert "rag-at-rest cycle" in text


def test_roadmap_records_the_rag_at_rest_cycle():
    text = _read("ROADMAP_POST_S183.md")
    assert "rag-at-rest cycle" in text
    assert "rs-01" in text
    # The cycle describes the application-layer encrypt-before-upsert direction.
    assert "encrypt" in text
    assert "aes-256-gcm" in text
