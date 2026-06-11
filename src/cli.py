"""
CLI entry point for the RAG knowledge base.

Usage:
    python -m src.cli index              # Index documents in data/raw/
    python -m src.cli query "问题"       # Query the knowledge base
    python -m src.cli interactive         # Interactive Q&A session
"""

import argparse
import logging
import sys

from src.config import LOG_LEVEL
from src.pipeline import RAGPipeline

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_index():
    """Index all documents in data/raw/."""
    pipeline = RAGPipeline()
    n = pipeline.index()
    if n > 0:
        print(f"✓ Indexed {n} chunks.")
    else:
        print("No documents indexed. Put PDF/TXT files in data/raw/.")


def cmd_query(question: str):
    """Answer a single question."""
    if not question:
        print("Error: Please provide a question.")
        sys.exit(1)

    pipeline = RAGPipeline()
    result = pipeline.query(question)

    print(f"\n{'='*60}")
    print(f"Q: {result['question']}")
    print(f"{'='*60}")
    print(f"A: {result['answer']}")
    print(f"\n--- Sources ({len(result['sources'])} chunks) ---")
    for s in result["sources"]:
        print(f"  [{s['source']}] score={s['score']}")
        print(f"  {s['text_preview']}...")
        print()


def cmd_interactive():
    """Start an interactive Q&A session."""
    pipeline = RAGPipeline()
    print("RAG Knowledge Base — Interactive Mode")
    print("Type 'exit' or 'quit' to leave.")
    print()

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break

        result = pipeline.query(question)
        print(f"\nA: {result['answer']}\n")


def main():
    parser = argparse.ArgumentParser(
        description="RAG Knowledge Base — CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("index", help="Index documents in data/raw/")

    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("question", nargs="?", default="", help="Your question")

    subparsers.add_parser("interactive", help="Interactive Q&A session")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index()
    elif args.command == "query":
        cmd_query(args.question)
    elif args.command == "interactive":
        cmd_interactive()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
