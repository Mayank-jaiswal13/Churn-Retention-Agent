"""RAG layer: TF-IDF retrieval over the retention playbook and support tickets.

TF-IDF keeps the project runnable with no downloads. To upgrade to dense retrieval,
swap the vectorizer for sentence-transformers embeddings + FAISS/Chroma; the
Retriever interface stays the same.
"""
import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent


class Retriever:
    def __init__(self):
        self.chunks = self._load_playbook()
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vec.fit_transform([c["title"] + " " + c["text"] for c in self.chunks])
        self.tickets = self._load_tickets()

    @staticmethod
    def _load_playbook():
        chunks = []
        for path in sorted((ROOT / "knowledge_base").glob("*.md")):
            for part in re.split(r"\n## ", path.read_text()):
                part = part.strip()
                if not part or part.startswith("# "):
                    continue
                title, _, body = part.partition("\n")
                chunks.append({"source": path.name, "title": title.strip("# ").strip(),
                               "text": body.strip()})
        return chunks

    @staticmethod
    def _load_tickets():
        path = ROOT / "data" / "support_tickets.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def search_playbook(self, query: str, k: int = 3):
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        top = sims.argsort()[::-1][:k]
        return [{**self.chunks[i], "score": round(float(sims[i]), 3)} for i in top if sims[i] > 0]

    def search_tickets(self, customer_id: str, query: str = "", k: int = 5):
        mine = [t for t in self.tickets if t["customer_id"] == customer_id]
        if query and len(mine) > 1:
            v = TfidfVectorizer(stop_words="english")
            try:
                m = v.fit_transform([t["text"] for t in mine] + [query])
                sims = cosine_similarity(m[-1], m[:-1])[0]
                mine = [mine[i] for i in sims.argsort()[::-1]]
            except ValueError:
                pass
        return mine[:k]
