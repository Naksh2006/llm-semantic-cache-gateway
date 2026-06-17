"""CLI tool for benchmarking semantic similarity between query pairs.

Usage examples:
  python -m benchmarks.sim_cli "How do I reset my password?" "I forgot my password"
  python -m benchmarks.sim_cli --threshold 0.88 --model BAAI/bge-small-en-v1.5 "query A" "query B"

This tool is useful during development to:
  • Calibrate the DEFAULT_SIMILARITY_THRESHOLD.
  • Compare embedding models side-by-side.
  • Spot-check whether two prompts would result in a cache hit.
"""

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the similarity CLI."""
    parser = argparse.ArgumentParser(
        prog="sim_cli",
        description="Compute cosine similarity between two queries using local embeddings.",
    )
    parser.add_argument(
        "query_a",
        type=str,
        help="First query string.",
    )
    parser.add_argument(
        "query_b",
        type=str,
        help="Second query string.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="FastEmbed model name (default: BAAI/bge-small-en-v1.5).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.92,
        help="Similarity threshold to compare against (default: 0.92).",
    )
    return parser


def main() -> None:
    """Entry-point — parse args and run similarity comparison."""
    parser = build_parser()
    args = parser.parse_args()

    # TODO: Load embedding model, compute cosine similarity, print results.
    print(f"Query A   : {args.query_a}")
    print(f"Query B   : {args.query_b}")
    print(f"Model     : {args.model}")
    print(f"Threshold : {args.threshold}")
    print("(similarity computation not yet implemented)")
    sys.exit(0)


if __name__ == "__main__":
    main()
