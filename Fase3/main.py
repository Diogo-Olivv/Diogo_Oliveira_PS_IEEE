"""CLI interativo do RAG."""
import argparse

from rag import RAGPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--closed_book", action="store_true")
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    rag = RAGPipeline(top_k=args.k)
    closed_book = args.closed_book
    k = args.k

    print("=" * 60)
    print("RAG Wikipedia PT-BR. Ctrl+C ou :q para sair")
    print(f"modo: {'closed-book' if closed_book else 'RAG'} | top_k={k}")
    print("=" * 60)

    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not q:
            continue
        if q in (":q", ":quit", ":exit"):
            break
        if q == ":cb":
            closed_book = not closed_book
            print(f"  modo: {'closed-book' if closed_book else 'RAG'}")
            continue
        if q.startswith(":k "):
            try:
                k = int(q.split()[1]); print(f"  top_k: {k}")
            except (IndexError, ValueError):
                print("  uso: :k <int>")
            continue

        result = rag.ask(q, k=k, closed_book=closed_book)
        print(f"\nresposta: {result['answer']}")
        if result["sources"]:
            print("\nfontes:")
            for s in result["sources"]:
                print(f"  {s['title']:40s} {s['score']:.3f}")


if __name__ == "__main__":
    main()
