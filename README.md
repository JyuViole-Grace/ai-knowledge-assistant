# AI Knowledge Assistant

A local, multilingual Retrieval-Augmented Generation (RAG) assistant that
answers questions about your documents and returns **verifiable citations**:
every quote it shows is checked against the original text and comes with its
exact character offsets. When no verified evidence exists, it says so instead
of guessing.

The project is also a case study in **measuring a RAG system honestly**: it ships
a retrieval benchmark with ablations, a strict end-to-end grader, and a written
list of the failure modes that the measurements uncovered.

## Highlights

- **Fully local.** Ollama + sentence-transformers; no data leaves the machine.
  Runs on an 8 GB GPU with `qwen2.5:7b`.
- **Hybrid retrieval:** multilingual dense embeddings + BM25, fused with
  Reciprocal Rank Fusion, then reranked by a CrossEncoder.
- **Verified quotes.** The LLM proposes quotes; the application maps each one
  back to the source text (tolerating whitespace, case, dash and quote-mark
  differences) and rejects fabricated or truncated ones. The answer is built
  only from the surviving verbatim quotes.
- **Safe abstention** when nothing is verified.
- **Ingestion that fits real files:** TXT, Markdown, PDF, DOCX; hyphenated line
  breaks and page numbers in old PDFs are repaired; chunks never split a word.
- **Interfaces:** CLI and a FastAPI service (path sandbox, optional API key).
- **Evaluation built in:** unit tests, retrieval metrics (Recall@k, MRR) with
  ablations, strict end-to-end grading against gold passages, and
  must-abstain tests.

## Results

These numbers come from a case study on two public-domain texts: a Russian
translation of Machiavelli's *The Prince* plus *Discourses on Livy* (835k
characters, 1030 chunks) and the original 16th-century Italian text of *Il
Principe* (177k characters, 221 chunks). **The texts and question files are not
included in this repository** (the Russian translation's licence is unclear);
the commands below work the same on any documents you provide.

Setup: Windows, 8 GB GPU, Ollama with `qwen2.5:7b`, 12 questions per language.

### Retrieval: each component earns its place

| Mode (Russian) | R@1 | R@3 | R@5 | R@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense only | 33% | 42% | 67% | 67% | 0.415 |
| BM25 only | 17% | 50% | 58% | 67% | 0.336 |
| hybrid (RRF) | 8% | 58% | 67% | 83% | 0.340 |
| **hybrid + rerank** | **75%** | **100%** | **100%** | **100%** | **0.861** |

| Mode (Italian) | R@1 | R@3 | R@5 | R@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense only | 8% | 17% | 33% | 42% | 0.174 |
| BM25 only | 17% | 25% | 42% | 50% | 0.244 |
| hybrid (RRF) | 8% | 50% | 50% | 58% | 0.292 |
| **hybrid + rerank** | **17%** | **50%** | **58%** | **58%** | **0.340** |

- The reranker is the biggest single gain (Russian R@1 8% → 75%).
- On the Italian text plain BM25 beats the dense embeddings: a general
  multilingual MiniLM does not know 16th-century spelling. That is why hybrid
  search is used.
- Italian recall is the weak spot. Three of the twelve questions are never
  retrieved by any mode, and the reranker can push a good chunk out of the top
  10 (Q9 is rank 2 after fusion). One cause is morphology: a query word such as
  "adulazione" does not match "adulatori" in a plain BM25 index (see `--bm25-prefix` below).

### Sensitivity: candidate pool and reranker

Same questions, hybrid + rerank, only the setting changes (no LLM involved):

| Setting | RU R@1 | RU R@5 | RU MRR | IT R@5 | IT R@10 | IT MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **default** (`--retrieve-k 18`, mMARCO MiniLM reranker) | 75% | 100% | 0.861 | 58% | 58% | 0.340 |
| `--retrieve-k 40` | 67% | 100% | 0.799 | 75% | 75% | 0.371 |
| `--retrieve-k 40` + `BAAI/bge-reranker-v2-m3` | not run | not run | not run | 58% | 83% | 0.407 |
| `--bm25-prefix 5` (crude stemming, `--retrieve-k 18`) | not run | not run | not run | 58% | 67% | 0.361 |

- A larger candidate pool is **corpus-dependent**. It rescued two Italian
  questions (R@5 58% → 75%) but slightly lowered the Russian score, where the
  *Discourses* add topical distractors that the reranker sometimes prefers.
- The larger bge reranker (2.3 GB) found two more Italian questions (R@10 83%)
  but ranked several hits lower, so it is not clearly better.
- Crude stemming for BM25 (`--bm25-prefix 5`) fixed the morphology miss ("adulazione"
  vs "adulatori": not found → rank 1) and lifted hybrid search on Italian from
  R@10 58% to 75% (MRR 0.292 → 0.372); BM25 alone got slightly worse on MRR
  (0.244 → 0.225). It has not been measured on Russian, so it stays off by
  default.
- With 12 questions per language none of these differences is conclusive, so
  the defaults stay as they are and both knobs are exposed as options to tune
  per corpus.

### End to end: strict grading

"Grounded" means at least one verified evidence quote overlaps a gold passage.
Precision is the share of all selected quotes that do.

| | Russian | Italian |
| --- | ---: | ---: |
| Grounded answers | **10/12 (83%)** | **2/12 (17%)** |
| Mean evidence precision | 28% | 6% |
| Latency per question (`--workers 4`) | ~28 s | ~23 s |

Off-topic questions (sport, cooking, chemistry, geography; in Russian, English
and Italian): **5/5 abstained.**

> Measurement notes: the retrieval tables and the abstention result were
> measured on the current pipeline. The Russian end-to-end number was measured
> just before the answer prompt was restricted to verbatim quotes; Italian was
> repeated afterwards with `--max-evidence 3` and scored 1/12 and 2/12
> (precision 4-12%). A one-to-two question difference between identical runs is
> noise on a 12-question set.

### What the measurements showed

| Run (strict grading, grounded answers) | Russian | Italian |
| --- | ---: | ---: |
| 1. One batched verification prompt, `--workers 4` | 8/12 | 1/12 |
| 2. + word-aligned chunks, `--workers 1` | 5/12 | 1/12 |
| 3. + one verification call per candidate | **10/12** | **2/12** |

- The batched verification prompt was the main cause of lost answers: it kept
  exactly one candidate in most questions and dropped correct quotes. Judging
  every candidate in its own short call fixed it (Russian 5 → 10).
- 69% of Russian and 59% of Italian chunks used to start in the middle of a
  word, and 119 of 120 PDF pages had their page number inside the text. Quotes
  copied across those spots could never match exactly; both are fixed.
- The fix exposed the opposite problem: the 7B judge answers `relevant=true`
  for nearly everything, so precision is low and weakly related quotes reach
  the answer. `--max-evidence 3` roughly doubled Italian precision (6% → ~12%)
  without changing the number of grounded answers.
- Italian is limited by the 7B model: relevant passages are retrieved but not
  extracted or judged reliably. A larger verifier (`--verify-model`) was not
  tried because it does not fit in 8 GB.

### Known failure modes

Documented from real runs:

- **A heading can become "verified evidence".** For "is it better to be loved
  or feared?" the Italian run kept the Latin chapter title *"et an sit melius
  amari quam timeri, vel e contra"* and dropped the sentence that answers it,
  so the assistant gave the opposite of the book's conclusion, with a valid
  citation. Quote verification proves a quote exists, not that it asserts
  anything.
- **A setup sentence can pass as the answer.** For "how should a prince treat
  promises?" a quote that only introduces the topic (*"how praiseworthy it is
  to keep faith, everyone understands"*) is a gold hit, but the book's actual
  point comes next. Strict grading measures grounding, not correctness.
- **Fragments.** A 35-character fragment from a different work in the same
  file was used to answer a question about promises.
- **Lenient relevance judging** by the small model (see above).

## Architecture

```text
Documents
   ↓
Text extraction, PDF repair (hyphenation, page numbers)
   ↓
Word-aligned chunks with overlap
   ↓
Embeddings + BM25  →  Reciprocal Rank Fusion  →  CrossEncoder rerank
   ↓
LLM quote extraction (parallel, one call per chunk)
   ↓
Exact quote recovery against the source text (offsets)
   ↓
LLM support/relevance check (one short call per candidate)
   ↓
Answer generated from the verbatim quotes only, with validated [E1] citations
```

## Quick start

Requirements: Python 3.10+, [Ollama](https://ollama.com).

```bash
git clone https://github.com/<your-user>/ai-knowledge-assistant.git
cd ai-knowledge-assistant
python -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull qwen2.5:7b
```

The embedding and reranker models download on the first run.

Put your files (`.txt`, `.md`, `.pdf`, `.docx`) in `documents/` (git ignores its
contents) or point `--docs` at any file or folder:

```bash
# interactive
python rag_assistant.py --docs documents

# one question
python rag_assistant.py --docs path/to/your/files --question "What does the contract say about termination?"
```

### Options

| Option | Purpose |
| --- | --- |
| `--model`, `--verify-model` | Ollama model for the pipeline / for the support check. A model must be installed first (`ollama pull <name>`); otherwise the run stops with a clear error. |
| `--workers N` | Parallel LLM calls. Use `1` for reproducible evaluation runs. |
| `--num-ctx N` | Ollama context window (default 8192; lower it on small GPUs). |
| `--max-evidence N` | Quotes used for the answer (default 5; 3 gives higher precision). |
| `--retrieve-k`, `--rerank-k` | Candidates passed to the reranker / kept after it. |
| `--bm25-prefix N` | Crude stemming for BM25: cut every token to N characters (try 5). Helps inflected languages. |
| `--embedding-model`, `--reranker-model` | Swap retrieval models (e.g. `BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`). |
| `--no-reranker`, `--no-cache`, `--quiet` | Disable reranking / cache, reduce logging. |

## Evaluation

```bash
pip install -r requirements-dev.txt
pytest -q                                   # unit tests, seconds, no models needed
```

The unit tests use fakes for the embedding model, reranker and Ollama. They
cover quote recovery, PDF repair, chunking, retrieval fusion, per-candidate
verification (including malformed and missing verdicts), citation validation,
the Ollama client, the API path sandbox and the metrics.

To evaluate on your own documents you write two small JSON files.

**Gold passages** (for retrieval metrics and strict grading). Copy, from the
document, short phrases that contain the answer; matching tolerates differences
in whitespace, case, dashes and quote marks, and `doc` is matched by file name:

```json
{"questions": [
  {"question": "When can the contract be terminated?",
   "doc": "contract.pdf",
   "gold": ["Either party may terminate this agreement with thirty days written notice."]}
]}
```

**Test cases** (end to end). Use `expected_sources` and `must_include` for
lenient checks, and `expect_abstain` for questions the documents cannot answer:

```json
{"questions": [
  {"question": "When can the contract be terminated?", "expected_sources": ["contract.pdf"]},
  {"question": "Who won the 2018 football World Cup?", "expect_abstain": true}
]}
```

```bash
# Retrieval: Recall@k / MRR in four modes, plus first-hit rank per question (no LLM)
python rag_assistant.py --docs documents --retrieval-eval my_gold.json

# End to end, strict: evidence must overlap a gold passage (needs Ollama)
python rag_assistant.py --docs documents --tests my_tests.json --gold my_gold.json --workers 1

# Without --gold only citation and source are checked (lenient)
python rag_assistant.py --docs documents --tests my_tests.json
```

- A retrieved chunk is a retrieval hit when it covers at least half of a gold
  phrase. Gold files are validated on load: a phrase missing from its document
  fails loudly instead of skewing the metrics.
- Without `--gold` a case only needs a citation and the right source (lenient:
  an irrelevant quote can pass). With `--gold` it must be grounded.
- Keep the gold list short but not too narrow: valid evidence elsewhere in the
  text counts as a miss, so the strict score is a conservative lower bound.
  Write the gold file *before* looking at the system's answers.

## API

```bash
uvicorn app.api:app --reload      # docs at http://127.0.0.1:8000/docs
```

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Service and index status (always open). |
| `POST` | `/index` | Builds an in-memory index. |
| `POST` | `/ask` | Answers a question with verified evidence. |

| Variable | Meaning |
| --- | --- |
| `RAG_DOCS_ROOT` | Directory clients may index (default `documents`). `docs_path` is relative to it; anything outside is rejected with `403`. |
| `RAG_API_KEY` | If set, `/index` and `/ask` require it in the `X-API-Key` header. |
| `OLLAMA_URL` | Ollama server (default `http://localhost:11434`). |

```bash
curl -X POST http://127.0.0.1:8000/index -H "Content-Type: application/json" -d '{"docs_path": "."}'
curl -X POST http://127.0.0.1:8000/ask   -H "Content-Type: application/json" \
     -d '{"question": "Why are mercenary armies unreliable?"}'
```

`/ask` holds no global lock while the LLM works, so requests run in parallel.
An unreachable or missing Ollama model answers `502`.

## Docker

```bash
docker compose up --build -d
docker compose exec ollama ollama pull qwen2.5:7b
curl -X POST http://127.0.0.1:8000/index -H "Content-Type: application/json" -d '{}'
```

`./documents` is mounted read-only; put your files there first. The compose setup
has not been verified end to end yet.

## Troubleshooting

- **`Ollama model '…' was not found`** — run the `ollama pull <model>` command
  the message shows.
- **Slow or out of GPU memory** — try `--num-ctx 4096` and `--workers 2`.
- **`unauthenticated requests to the HF Hub`** — harmless; set `HF_TOKEN` to
  silence it.
- **First run is slow** — embeddings are computed once and cached in
  `.rag_cache/` (the cache is invalidated automatically when chunking changes).

## Project structure

```text
app/
  answer.py  api.py  assistant.py  cli.py  config.py  evaluation.py
  evidence.py  ingestion.py  llm.py  quotes.py  retrieval.py
  retrieval_eval.py  schemas.py  security.py
documents/          put your files here (contents are ignored by git)
tests/              pytest suite (no models, Ollama or documents needed)
.github/workflows/ci.yml   Dockerfile   docker-compose.yml
requirements.txt  requirements-dev.txt  pytest.ini  .env.example  LICENSE
```

## Limitations

- Scanned PDFs need OCR; they contain no extractable text.
- A 7B model needs about a dozen LLM calls per question (20–30 s on an 8 GB GPU)
  and is noticeably weaker on archaic Italian than on Russian.
- The support/relevance check is an LLM judgement and can be lenient or wrong
  (see the failure modes above).
- Only verbatim quotes reach the answer step, but the validator checks that cited
  ids exist, not that each sentence is supported by its quote.
- BM25 uses plain tokenization by default (no stemming), so it misses inflected
  forms; `--bm25-prefix` is a crude workaround, not a real stemmer.
- Chapter headings and other non-assertive text can be extracted as evidence.
- Abstention messages exist for Russian, Ukrainian and English only.
- The index lives in memory; large collections would need a vector database.

## Roadmap

- [x] Hybrid retrieval, exact quote verification, abstention.
- [x] CLI and FastAPI service with path sandbox and optional API key.
- [x] Unit tests; retrieval benchmark with ablations; strict grading;
      must-abstain tests; works on any document set.
- [ ] Confirm the GitHub Actions workflow is green after the first push.
- [ ] Verify the Docker setup end to end.
- [ ] Measure `--bm25-prefix` on the Russian case study (Italian: hybrid R@10 58% → 75%).
- [ ] Compare rerankers on the Russian case study.
- [ ] Try a larger verification model (needs more than 8 GB of GPU memory).
- [ ] Filter headings out of the evidence; check support per answer sentence.
- [ ] OCR for scanned PDFs, vector database, web interface.

## License

MIT, see [LICENSE](LICENSE).
