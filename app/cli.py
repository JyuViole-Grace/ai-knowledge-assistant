import argparse
import sys

from .assistant import RAGAssistant
from .config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_RERANKER_MODEL,
    Config,
    setup_logging,
)
from .evaluation import run_tests
from .retrieval_eval import run_retrieval_eval


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Knowledge Assistant")

    parser.add_argument("--docs", default="documents")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--verify-model",
        default="",
        help="Ollama model for the support/relevance check (default: --model).",
    )
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--question")
    parser.add_argument("--tests", help="End-to-end smoke tests (needs Ollama).")
    parser.add_argument(
        "--gold",
        metavar="GOLD_FILE",
        help="Strict grading for --tests: evidence must overlap a gold passage.",
    )
    parser.add_argument(
        "--retrieval-eval",
        metavar="GOLD_FILE",
        help="Recall@k / MRR against gold passages (no LLM needed).",
    )
    parser.add_argument("--cache-dir", default=".rag_cache")
    parser.add_argument(
        "--workers",
        type=int,
        default=Config.workers,
        help="Parallel LLM calls during evidence extraction.",
    )
    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=Config.retrieve_k,
        help="Candidates passed to the reranker.",
    )
    parser.add_argument(
        "--rerank-k",
        type=int,
        default=Config.rerank_k,
        help="Chunks kept after reranking (evidence is extracted from these).",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=Config.num_ctx,
        help="Ollama context window in tokens (lower it on small GPUs).",
    )
    parser.add_argument(
        "--max-evidence",
        type=int,
        default=Config.max_evidence,
        help="Maximum number of evidence quotes used for the answer.",
    )
    parser.add_argument(
        "--bm25-prefix",
        type=int,
        default=Config.bm25_prefix,
        help="Crude stemming: cut BM25 tokens to N characters (0 = off, try 5).",
    )
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    return parser


def main() -> int:
    args = build_parser().parse_args()
    setup_logging()

    config = Config(
        docs=args.docs,
        ollama_url=args.ollama_url,
        model=args.model,
        verify_model=args.verify_model,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        cache_dir=args.cache_dir,
        use_reranker=not args.no_reranker,
        use_cache=not args.no_cache,
        workers=args.workers,
        num_ctx=args.num_ctx,
        max_evidence=args.max_evidence,
        bm25_prefix=args.bm25_prefix,
        retrieve_k=args.retrieve_k,
        rerank_k=args.rerank_k,
        debug=not args.quiet,
    )

    try:
        assistant = RAGAssistant(config)

        if args.question:
            assistant.ask(args.question)

        elif args.tests:
            run_tests(assistant, args.tests, args.gold)

        elif args.retrieval_eval:
            run_retrieval_eval(assistant, args.retrieval_eval)

        else:
            print("AI Knowledge Assistant")
            print("Enter a question or type /exit.")

            while True:
                try:
                    question = input("\nYou: ").strip()
                except (KeyboardInterrupt, EOFError):
                    print("\nStopped.")
                    break

                if question.lower() in {"/exit", "exit", "quit", "/quit"}:
                    break

                if question:
                    assistant.ask(question)

        return 0

    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
