"""Baixa a Wikipedia PT, divide em chunks, gera embeddings e salva no FAISS."""
import argparse
import json
import pickle
from pathlib import Path

import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_articles", type=int, default=2000)
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    print(f"[1/4] wiki PT streaming n={args.n_articles} ...")
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.pt",
        split="train", streaming=True,
    )
    articles = []
    for i, row in enumerate(ds):
        if i >= args.n_articles:
            break
        articles.append({"title": row["title"], "text": row["text"], "url": row["url"]})

    print(f"[2/4] chunking size={args.chunk_size} overlap={args.overlap} ...")
    chunks, meta = [], []
    for art in tqdm(articles):
        for j, ch in enumerate(chunk_text(art["text"], args.chunk_size, args.overlap)):
            chunks.append(ch)
            meta.append({"title": art["title"], "url": art["url"], "chunk_id": j})
    print(f"   total chunks = {len(chunks)}")

    print(f"[3/4] embeddings {args.embed_model} ...")
    model = SentenceTransformer(args.embed_model)
    emb = model.encode(
        chunks,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    print(f"[4/4] FAISS dim={emb.shape[1]} ...")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    faiss.write_index(index, str(DATA_DIR / "wiki.faiss"))
    with open(DATA_DIR / "chunks.pkl", "wb") as f:
        pickle.dump({"chunks": chunks, "meta": meta}, f)
    with open(DATA_DIR / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print(f"\nConcluido em {DATA_DIR}")
    print(f"   wiki.faiss com {len(chunks)} vetores")
    print(f"   chunks.pkl")


if __name__ == "__main__":
    main()
