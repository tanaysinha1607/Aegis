from rag.retriever import FaissNarrativeRetriever


def test_faiss_retriever_build_and_search():
    narratives = [
        {"text": "High amount and new device fraud pattern alpha.", "risk_label": 1},
        {"text": "Benign grocery spend baseline beta.", "risk_label": 0},
        {"text": "Velocity spike across merchants gamma.", "risk_label": 1},
    ]
    r = FaissNarrativeRetriever(embedding_model_name="sentence-transformers/all-MiniLM-L6-v2")
    r.build_index(narratives)
    hits = r.search("unusual amount and new device", top_k=2)
    assert len(hits) >= 1
    assert isinstance(hits[0][0], str)
