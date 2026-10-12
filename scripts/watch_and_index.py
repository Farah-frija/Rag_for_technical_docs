"""
scripts/watch_and_index.py
-----------------------------
Optional convenience script: watch a directory for new PDFs and run the
full index -> (auto) retrieve pipeline on each one automatically, with zero
manual intervention. This is the fully-automatic end of the "retrieval
should be launched whether manually or automatically" spectrum -- the CLI's
`manual-rag index` command covers the manual end (you run it yourself, once
per PDF, and it launches retrieval when done).

Usage:
    python scripts/watch_and_index.py ./incoming_manuals ./indexed --interval 10

Every *.pdf that appears in `watch_dir` gets indexed to
`<output_dir>/<pdf_stem>.json`, then a smoke-test query is run against it
via the same `IndexingPipeline.run(on_complete=...)` hook the CLI uses --
so this script and `manual-rag index` share the exact same completion
semantics, just triggered differently (a filesystem poll vs. a manual
command).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Set

from manual_rag.config import get_settings
from manual_rag.indexing.pipeline import IndexingPipeline
from manual_rag.logging_config import configure_logging, get_logger
from manual_rag.retrieval.factory import get_retriever

logger = get_logger("watch_and_index")


def _smoke_test_hook(settings, strategy: str):
    def hook(index_path, index):
        retriever = get_retriever(index, settings, strategy=strategy)
        for q in settings.indexing.smoke_test_queries:
            result = retriever.retrieve(q)
            logger.info(f"[smoke-test] Q: {q!r} -> confidence={result.confidence} answer={result.answer[:200]!r}")
    return hook


def watch(watch_dir: Path, output_dir: Path, interval: float, retriever: str) -> None:
    settings = get_settings()
    pipeline = IndexingPipeline(settings)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed: Set[str] = set()
    logger.info(f"Watching '{watch_dir}' every {interval}s for new PDFs ...")

    while True:
        for pdf_path in sorted(watch_dir.glob("*.pdf")):
            if pdf_path.name in processed:
                continue
            output_path = output_dir / f"{pdf_path.stem}.json"
            logger.info(f"New PDF detected: {pdf_path.name} -> indexing to {output_path}")
            try:
                pipeline.run(
                    pdf_path=str(pdf_path),
                    output_path=str(output_path),
                    on_complete=_smoke_test_hook(settings, retriever),
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Indexing failed for {pdf_path.name}: {exc}")
            finally:
                processed.add(pdf_path.name)
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a directory and auto-index new PDFs.")
    parser.add_argument("watch_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--interval", type=float, default=10.0, help="Poll interval in seconds.")
    parser.add_argument("--retriever", default="tree", choices=["tree", "vanilla", "page_index"])
    args = parser.parse_args()

    configure_logging()
    watch(args.watch_dir, args.output_dir, args.interval, args.retriever)


if __name__ == "__main__":
    main()
