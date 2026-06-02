"""RAGPipeline: recuperacao e geracao."""
import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DATA_DIR = Path(__file__).parent / "data"

PROMPT_TEMPLATE = """Responda à pergunta apenas com base no contexto abaixo. \
Se a resposta não estiver no contexto, diga "Não sei".

Contexto:
{context}

Pergunta: {question}
Resposta:"""


class RAGPipeline:
    def __init__(
        self,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        gen_model: str = "google/flan-t5-base",
        top_k: int = 4,
        max_new_tokens: int = 256,
    ):
        if not (DATA_DIR / "wiki.faiss").exists():
            raise FileNotFoundError(
                "Index não encontrado. Execute python ingest.py primeiro."
            )

        print("carregando index ...")
        self.index = faiss.read_index(str(DATA_DIR / "wiki.faiss"))
        with open(DATA_DIR / "chunks.pkl", "rb") as f:
            blob = pickle.load(f)
        self.chunks = blob["chunks"]
        self.meta = blob["meta"]

        print(f"carregando embedder {embed_model} ...")
        self.embedder = SentenceTransformer(embed_model)

        print(f"carregando LLM {gen_model} ...")
        self.tok = AutoTokenizer.from_pretrained(gen_model)
        self.llm = AutoModelForSeq2SeqLM.from_pretrained(gen_model)

        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        print(f"pronto. {len(self.chunks)} chunks indexados.\n")

    def retrieve(self, query: str, k: int | None = None):
        k = k or self.top_k
        q = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, idxs = self.index.search(q, k)
        hits = []
        for score, i in zip(scores[0], idxs[0]):
            if i == -1:
                continue
            hits.append({
                "score": float(score),
                "text": self.chunks[i],
                "title": self.meta[i]["title"],
                "url": self.meta[i]["url"],
            })
        return hits

    def generate(self, question: str, context: str) -> str:
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        inputs = self.tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
        out = self.llm.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            num_beams=4,
        )
        return self.tok.decode(out[0], skip_special_tokens=True)

    def ask(self, question: str, k: int | None = None, closed_book: bool = False):
        if closed_book:
            answer = self.generate(question, context="sem contexto")
            return {"answer": answer, "sources": [], "closed_book": True}

        hits = self.retrieve(question, k)
        context = "\n\n\n\n".join(
            f"[{h['title']}]\n{h['text']}" for h in hits
        )
        answer = self.generate(question, context)
        return {
            "answer": answer,
            "sources": [
                {"title": h["title"], "url": h["url"], "score": h["score"]}
                for h in hits
            ],
            "closed_book": False,
        }
