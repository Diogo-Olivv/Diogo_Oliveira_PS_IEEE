"""Comparacao qualitativa entre RAG e closed-book."""
from rag import RAGPipeline

QUESTIONS = [
    "Quem foi Dom Pedro II?",
    "O que é o teorema de Pitágoras?",
    "Qual a capital do estado do Amazonas?",
    "Quando foi proclamada a República no Brasil?",
    "O que é fotossíntese?",
]


def main():
    rag = RAGPipeline()
    for q in QUESTIONS:
        print("=" * 80)
        print(f"Q: {q}\n")
        cb = rag.ask(q, closed_book=True)
        rg = rag.ask(q, closed_book=False)
        print(f"[closed-book] {cb['answer']}")
        print(f"[RAG]         {rg['answer']}")
        print(f"fontes: {[s['title'] for s in rg['sources']]}")
    print("=" * 80)


if __name__ == "__main__":
    main()
