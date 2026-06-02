# RAG Wikipedia PT-BR local

Pipeline RAG simples sobre `wikimedia/wikipedia` PT-BR.
Atividade Trainee CIS 2026, IEEE CIS UnB.

## Stack

- **Retriever**: `sentence-transformers/all-MiniLM-L6-v2` com FAISS, cosseno via inner product
- **Generator**: `google/flan-t5-base`
- **Dataset**: `wikimedia/wikipedia` config `20231101.pt` em modo streaming

## Estrutura

```
.
├── ingest.py        baixa wiki, gera chunks, embeddings e FAISS
├── rag.py           classe RAGPipeline
├── main.py          CLI interativo
├── eval.py          RAG versus closed-book
├── data/            index e metadados gerados por ingest.py
└── requirements.txt
```

## Setup Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
# construir o index, comece com poucos artigos
python ingest.py --n_articles 500

# CLI
python main.py

# comparacao RAG versus closed-book
python eval.py
```

### Comandos dentro do CLI

| comando | efeito              |
| ------- | ------------------- |
| `:cb`   | alterna closed-book |
| `:k 6`  | altera o top-k      |
| `:q`    | sair                |

## Analise do Chunk Size

```bash
python ingest.py --chunk_size 256 --overlap 32
mv data data_256
python ingest.py --chunk_size 512 --overlap 64
mv data data_512
```

Execute o eval em cada configuracao e compare os resultados.

Heuristicas:

- **256**: trechos precisos, porem podem cortar contexto necessario.
- **512**: mais contexto, com top-1 menos preciso.
- **overlap** de aproximadamente 10 a 15 por cento do chunk_size e um bom default.

## Observações

- `IndexFlatIP` tem complexidade O(N), serve pra até 1M chunks. Acima disso, utilize `IndexIVFFlat` ou HNSW.
- `flan-t5-base` tem desempenho limitado em portugues puro. Como melhoria, considere `unicamp-dl/ptt5-base-portuguese-vocab` ou Llama-3.

## Tempo aproximado (CPU)

| n_articles | chunks | ingest |  RAM |
| ---------: | -----: | -----: | ---: |
|        500 |     4k |  3 min | 1 GB |
|       2000 |    16k | 12 min | 2 GB |
|       5000 |    40k | 30 min | 4 GB |
