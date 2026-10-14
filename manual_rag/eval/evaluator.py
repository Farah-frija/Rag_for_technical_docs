from __future__ import annotations

import ast
import csv
import json
import os
import re
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from manual_rag.config import Settings
from manual_rag.core.interfaces import Retriever
from manual_rag.core.models import ManualIndex
from manual_rag.logging_config import get_logger

from manual_rag.utils.shared_helpers import resolve_cited_ids

logger = get_logger(__name__)

_WS_RE = re.compile(r"\s+")

# shapes:
#   flat                -> one list of ids
#   per_subquery        -> list of id lists (one per sub-query)
#   per_subquery_cited  -> same, but ids were written by the LLM: they are
#                          validated against the stage-3 ids of the same
#                          sub-query before being treated like any other ids
STAGE_KEY_MAP: Dict[str, List[Tuple[str, str, str]]] = {
    "tree": [
        ("stage1_ids", "post_embed_prune", "flat"),
        ("stage2_per_subquery", "post_llm_router", "per_subquery"),
        ("stage3_per_subquery", "post_cross_encoder", "per_subquery"),
    ],
    "vanilla": [
        ("stage2_per_subquery", "post_fusion", "per_subquery"),
        ("stage3_per_subquery", "post_cross_encoder", "per_subquery"),
        ("stage4_per_subquery", "post_llm_citation", "per_subquery_cited"),
    ],
    # layout stages depend on the step toggles: see get_stages()
    "layout": [
        ("stage1_dense_per_subquery", "post_dense", "per_subquery"),
        ("stage1_bm25_per_subquery", "post_bm25", "per_subquery"),
        ("stage2_per_subquery", "post_fusion", "per_subquery"),
        ("stage3_per_subquery", "post_cross_encoder", "per_subquery"),
        ("stage3b_per_subquery", "post_context_cut", "per_subquery"),  # NEW
        ("stage4_per_subquery", "post_llm_citation", "per_subquery_cited"),
    ],
}

_STAGE_METRIC_FIELDS = [
    "precision", "recall", "f1", "hit", "n_pred", "n_expected", "n_correct",
    "context_precision_mean", "subquery_precisions", "mrr",
]
_SUPPORTS_SPLIT_SUBQUERIES = {"tree", "layout"}
_CHUNK_BASED_RETRIEVERS = {"vanilla", "layout"}


def get_stages(retriever_name: str, settings: Settings) -> List[Tuple[str, str, str]]:
    """Stages to measure. For layout, drop the stages whose step is disabled
    (no dense -> no post_dense, no bm25 -> no post_bm25, no LLM -> no citations)."""
    stages = STAGE_KEY_MAP.get(retriever_name, [])
    if retriever_name != "layout":
        return stages
    r = settings.retrieval
    skip = set()
    if not r.use_dense:
        skip.add("post_dense")
    if not r.use_bm25:
        skip.add("post_bm25")
    if not r.use_llm_responder:
        skip.add("post_llm_citation")
    return [st for st in stages if st[1] not in skip]


def get_stage_k(retriever_name: str, settings: Settings) -> Dict[str, str]:
    """Configured k per stage, shown in the summary."""
    if retriever_name != "layout":
        return {}
    r = settings.retrieval
    fusion_cap = (r.top_n_dense if r.use_dense else 0) + (r.top_n_bm25 if r.use_bm25 else 0)
    return {
        "post_dense": str(r.top_n_dense),
        "post_bm25": str(r.top_n_bm25),
        "post_fusion": f"<={fusion_cap}",
        "post_cross_encoder": str(r.top_n_fusion),
        "post_context_cut": str(r.top_n_for_context),
        "post_llm_citation": f"<={r.top_n_for_context}",
    }


def get_timing_steps(retriever_name: str, settings: Settings) -> List[str]:
    """Timing steps recorded by the retriever (layout only), in pipeline order."""
    if retriever_name != "layout":
        return []
    r = settings.retrieval
    steps: List[str] = []
    if r.use_subquery_split:
        steps.append("subquery_split")
    if r.use_dense:
        steps += ["query_embedding", "dense_recall"]
    if r.use_bm25:
        steps.append("bm25_recall")
    steps.append("rrf_fusion")
    if r.use_reranker:
        steps.append("rerank")
    if r.use_llm_responder:
        steps.append("responder_llm")
    steps.append("retrieve_total")
    return steps


def _answer_cosine(embedder: Any, expected: str, final: str) -> float:
    """Cosine similarity between embeddings of the expected and final answers."""
    if embedder is None or not expected or not final:
        return float("nan")
    try:
        a = np.asarray(embedder.embed_one(expected), dtype=float)
        b = np.asarray(embedder.embed_one(final), dtype=float)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b / denom) if denom else float("nan")
    except Exception as exc:
        logger.warning(f"answer cosine failed: {exc}")
        return float("nan")


def _normalise(name: Optional[str]) -> str:
    if name is None:
        return ""
    n = name.strip().lower()
    n = _WS_RE.sub(" ", n)
    return n.rstrip(".")


def _parse_expected_sections(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    raw = str(raw).strip()
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
        return [str(x) for x in parsed] if isinstance(parsed, (list, tuple)) else [str(parsed)]
    except (ValueError, SyntaxError):
        return [s.strip().strip("'\"") for s in raw.strip("[]").split(",") if s.strip()]


def _ids_to_names(index: ManualIndex, node_ids: List[str]) -> List[str]:
    names = []
    for nid in node_ids:
        node = index.nodes.get(nid)
        if node is not None:
            names.append(node.content)
    return names


def _chunk_ids_to_section_names(
    chunk_ids: List[str],
    id_to_chunk: Dict[str, Dict[str, Any]],
) -> List[str]:
    names: List[str] = []
    for cid in chunk_ids:
        chunk = id_to_chunk.get(cid)
        if chunk is None:
            continue
        val = chunk.get("source_sections")
        if isinstance(val, list):
            names.extend(val)
        elif isinstance(val, str) and val:
            names.append(val)
    return names


def _resolve_names(
    index: ManualIndex,
    retriever_name: str,
    ids: List[str],
    id_to_chunk: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[str]:
    if retriever_name in _CHUNK_BASED_RETRIEVERS:
        return _chunk_ids_to_section_names(ids, id_to_chunk or {})
    return _ids_to_names(index, ids)


def context_precision_at_k(relevance_flags: List[bool]) -> float:
    total_relevant = sum(relevance_flags)
    if total_relevant == 0:
        return 0.0
    running_relevant = 0
    numerator = 0.0
    for k, is_relevant in enumerate(relevance_flags, start=1):
        if is_relevant:
            running_relevant += 1
            numerator += running_relevant / k
    return numerator / total_relevant


def mean_context_precision(subquery_precisions: List[float]) -> float:
    if not subquery_precisions:
        return 0.0
    return sum(subquery_precisions) / len(subquery_precisions)


def reciprocal_rank(relevance_flags: List[bool]) -> float:
    """1 / rank of the first relevant item, 0.0 if none is relevant."""
    for k, is_relevant in enumerate(relevance_flags, start=1):
        if is_relevant:
            return 1.0 / k
    return 0.0


def _ranked_relevance_flags(
    index: ManualIndex,
    retriever_name: str,
    ranked_ids: List[str],
    expected_norm: Set[str],
    id_to_chunk: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[bool]:
    flags: List[bool] = []
    for nid in ranked_ids:
        names = _resolve_names(index, retriever_name, [nid], id_to_chunk)
        norm_names = {_normalise(n) for n in names if _normalise(n)}
        flags.append(bool(norm_names & expected_norm))
    return flags


def _set_metrics(predicted_names: List[str], expected_names: List[str]) -> Dict[str, float]:
    pred_set = {_normalise(n) for n in predicted_names if _normalise(n)}
    exp_set = {_normalise(n) for n in expected_names if _normalise(n)}
    if not exp_set:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan"),
                "hit": float("nan"), "n_pred": len(pred_set), "n_expected": 0, "n_correct": 0}
    correct = pred_set & exp_set
    n_pred, n_exp, n_correct = len(pred_set), len(exp_set), len(correct)
    precision = n_correct / n_pred if n_pred else 0.0
    recall = n_correct / n_exp if n_exp else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "hit": 1.0 if n_correct else 0.0,
            "n_pred": n_pred, "n_expected": n_exp, "n_correct": n_correct}


def _per_subquery_stage_metrics(
    index: ManualIndex,
    retriever_name: str,
    per_subquery_ids: List[List[str]],
    expected_sections: List[str],
    id_to_chunk: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    expected_norm = {_normalise(n) for n in expected_sections if _normalise(n)}
    if not expected_norm:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan"),
                "hit": float("nan"), "n_pred": 0, "n_expected": 0, "n_correct": 0,
                "context_precision_mean": float("nan"), "subquery_precisions": [],
                "mrr": float("nan")}

    subqueries = per_subquery_ids or [[]]

    # Rank-aware metrics, kept per subquery (reported only, not used for precision/F1)
    flags_per_sq = [
        _ranked_relevance_flags(index, retriever_name, ids, expected_norm, id_to_chunk)
        for ids in subqueries
    ]
    subquery_precisions = [context_precision_at_k(f) for f in flags_per_sq]
    context_precision_mean = mean_context_precision(subquery_precisions)
    mrr = sum(reciprocal_rank(f) for f in flags_per_sq) / len(flags_per_sq)

    # Pooled (fusion) metrics: precision, recall and F1 all use the union across subqueries
    pooled_ids: Set[str] = set()
    for ids in subqueries:
        pooled_ids.update(ids)
    pooled_names = _resolve_names(index, retriever_name, sorted(pooled_ids), id_to_chunk)
    pooled_norm = {_normalise(n) for n in pooled_names if _normalise(n)}

    n_correct = len(pooled_norm & expected_norm)
    n_expected = len(expected_norm)
    n_pred = len(pooled_norm)

    precision = n_correct / n_pred if n_pred else 0.0
    recall = n_correct / n_expected if n_expected else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "hit": 1.0 if n_correct else 0.0,
        "n_pred": n_pred, "n_expected": n_expected, "n_correct": n_correct,
        "context_precision_mean": context_precision_mean,
        "subquery_precisions": [round(p, 4) for p in subquery_precisions],
        "mrr": mrr,
    }


def _pooled_predicted_names(
    index: ManualIndex,
    retriever_name: str,
    per_subquery_ids: List[List[str]],
    id_to_chunk: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[str]:
    pooled_ids: Set[str] = set()
    for ids in (per_subquery_ids or [[]]):
        pooled_ids.update(ids)
    return _resolve_names(index, retriever_name, sorted(pooled_ids), id_to_chunk)


@dataclass
class EvalRow:
    values: Dict[str, Any]


def evaluate_query(
    index: ManualIndex,
    settings: Settings,
    row: Dict[str, Any],
    retriever_name: str,
    split_subqueries: bool = True,
    id_to_chunk: Optional[Dict[str, Dict[str, Any]]] = None,
    retriever: Optional[Retriever] = None,
    embedder: Optional[Any] = None,
) -> Dict[str, Any]:
    query_id = row.get("id", "")
    category = str(row.get("category", "")).strip()
    question = str(row.get("question", "")).strip()
    expected_answer = str(row.get("expected_answer", "")).strip()
    expected_sections = _parse_expected_sections(row.get("sections"))

    result: Dict[str, Any] = {
        "id": query_id, "retriever": retriever_name, "category": category,
        "question": question, "expected_sections": expected_sections,
        "expected_answer": expected_answer, "error": "",
        "split_subqueries": split_subqueries,
    }

    try:

        if retriever_name in _SUPPORTS_SPLIT_SUBQUERIES:
            # layout: True defers to settings.retrieval.use_subquery_split
            flag = None if (retriever_name == "layout" and split_subqueries) else split_subqueries
            retrieval = retriever.retrieve(question, split_subqueries=flag)
        else:
            retrieval = retriever.retrieve(question)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        return result

    # Layout: stage 3 holds top_n_fusion chunks but the LLM only saw the first
    # top_n_for_context of them, so citations are validated against those.
    n_shown = settings.retrieval.top_n_for_context if retriever_name == "layout" else None

    for key, label, shape in get_stages(retriever_name, settings):
        raw = retrieval.stage_capture.get(key, [])
        if shape in ("per_subquery", "per_subquery_cited"):
            per_subquery_ids = raw or []

            if shape == "per_subquery_cited":
                # The LLM wrote these ids: keep only those that were actually
                # shown to it for the same sub-query (stage 3). After this
                # they are ordinary chunk ids, resolved like every other stage.
                ce_lists = retrieval.stage_capture.get("stage3_per_subquery") or []
                valid_lists: List[List[str]] = []
                n_unresolved = 0
                for i, cited in enumerate(per_subquery_ids):
                    shown = ce_lists[i] if i < len(ce_lists) else []
                    if n_shown is not None:
                        shown = shown[:n_shown]
                    valid, unresolved = resolve_cited_ids(cited, shown)
                    valid_lists.append(valid)
                    n_unresolved += len(unresolved)
                result[f"{label}_raw"] = per_subquery_ids
                result[f"{label}_unresolved"] = n_unresolved
                per_subquery_ids = valid_lists

            metrics = _per_subquery_stage_metrics(
                index, retriever_name, per_subquery_ids, expected_sections, id_to_chunk
            )
            names = _pooled_predicted_names(index, retriever_name, per_subquery_ids, id_to_chunk)
        else:
            ids = raw or []
            names = _resolve_names(index, retriever_name, ids, id_to_chunk)
            metrics = _set_metrics(names, expected_sections)
            exp_norm = {_normalise(n) for n in expected_sections if _normalise(n)}
            metrics["mrr"] = (
                reciprocal_rank(_ranked_relevance_flags(index, retriever_name, ids, exp_norm, id_to_chunk))
                if exp_norm else float("nan")
            )
        for m, v in metrics.items():
            result[f"{label}_{m}"] = v
        result[f"{label}_predicted_names"] = names

    final_answer = retrieval.answer
    predicted_unknown = (
        retrieval.confidence == "not_found" or not final_answer or final_answer.strip().lower() == "unknown"
    )
    result["final_answer"] = final_answer
    result["final_confidence"] = retrieval.confidence
    result["predicted_unknown"] = predicted_unknown
    result["num_subqueries"] = len(retrieval.subquery_results)
    result["subqueries_text"] = [r.subquery for r in retrieval.subquery_results]
    result["multi_hop_not_split"] = category.lower() == "multi-hop" and len(retrieval.subquery_results) <= 1
    result["tokens_total"] = sum((e.get("tokens") or {}).get("total_tokens") or 0 for e in retrieval.trace_entries)

    # Timings recorded by the retriever (layout only), summed over sub-queries
    timings = retrieval.stage_capture.get("timings_sec", {}) or {}
    for step, v in timings.items():
        result[f"time_{step}"] = v

    # cosine(expected_answer, final_answer); meaningless if the answer is just raw chunks
    answer_is_llm = not (retriever_name == "layout" and not settings.retrieval.use_llm_responder)
    result["answer_cosine"] = (
        _answer_cosine(embedder, expected_answer, final_answer) if answer_is_llm else float("nan")
    )

    return result


class IncrementalCsvWriter:
    def __init__(self, path: str, fieldnames: List[str], append: bool = False) -> None:
        file_exists = append and os.path.exists(path) and os.path.getsize(path) > 0
        self._fh = open(path, "a" if file_exists else "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames)
        if not file_exists:
            self._writer.writeheader()
        self._fh.flush()

    def write_row(self, row: Dict[str, Any]) -> None:
        flat = {}
        for field in self._writer.fieldnames:
            v = row.get(field, "")
            flat[field] = json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
        self._writer.writerow(flat)
        self._fh.flush()
        try:
            os.fsync(self._fh.fileno())
        except OSError:
            pass

    def close(self) -> None:
        self._fh.close()


def _load_detailed_csv(path: str) -> List[Dict[str, Any]]:
    """Reload a *_detailed.csv so aggregate() can run over every row
    that's ever been written to it, not just the rows from one call.
    Coerces the columns aggregate()/print_summary() use back to sane
    Python types (lists were JSON-encoded, bools became 'True'/'False'
    text, blanks became NaN)."""
    df = pd.read_csv(path)
    rows = df.to_dict(orient="records")
    for r in rows:
        if pd.isna(r.get("error")):
            r["error"] = ""
        if pd.isna(r.get("category")):
            r["category"] = ""
        pu = r.get("predicted_unknown")
        if isinstance(pu, str):
            r["predicted_unknown"] = pu.strip().lower() == "true"
        elif isinstance(pu, float) and pd.isna(pu):
            r["predicted_unknown"] = False
    return rows


def _build_csv_fields(retriever_name: str, settings: Settings) -> List[str]:
    fields = ["id", "retriever", "split_subqueries", "category", "question", "expected_sections", "expected_answer"]
    for _key, label, shape in get_stages(retriever_name, settings):
        for m in _STAGE_METRIC_FIELDS:
            fields.append(f"{label}_{m}")
        fields.append(f"{label}_predicted_names")
        if shape == "per_subquery_cited":
            # audit columns; without them the writer silently drops the values
            fields.append(f"{label}_raw")
            fields.append(f"{label}_unresolved")
    fields += [
        "num_subqueries", "subqueries_text", "multi_hop_not_split",
        "final_answer", "final_confidence", "predicted_unknown", "tokens_total",
        "answer_cosine",
    ]
    fields += [f"time_{s}" for s in get_timing_steps(retriever_name, settings)]
    fields += ["elapsed_sec", "error", "traceback"]
    return fields


def _nanmean(values: List[float]) -> Optional[float]:
    clean = [v for v in values if v is not None and v == v]
    return sum(clean) / len(clean) if clean else None


def aggregate(
    rows: List[Dict[str, Any]],
    stages: List[Tuple[str, str, str]],
    timing_steps: Optional[List[str]] = None,
) -> Dict[str, Any]:
    timing_steps = timing_steps or []

    def stage_block(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        block = {}
        for _, label, _shape in stages:
            block[label] = {
                "precision": _nanmean([r.get(f"{label}_precision") for r in subset]),
                "recall": _nanmean([r.get(f"{label}_recall") for r in subset]),
                "f1": _nanmean([r.get(f"{label}_f1") for r in subset]),
                "hit_rate": _nanmean([r.get(f"{label}_hit") for r in subset]),
                "context_precision": _nanmean([r.get(f"{label}_context_precision_mean") for r in subset]),
                "mrr": _nanmean([r.get(f"{label}_mrr") for r in subset]),
                "avg_n_pred": _nanmean([r.get(f"{label}_n_pred") for r in subset]),
                "n_queries": len(subset),
            }
        return block

    def extras_block(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "answer_cosine": _nanmean([r.get("answer_cosine") for r in subset]),
            "avg_elapsed_sec": _nanmean([r.get("elapsed_sec") for r in subset]),
            "timing_by_step": {
                s: _nanmean([r.get(f"time_{s}") for r in subset]) for s in timing_steps
            },
        }

    summary: Dict[str, Any] = {
        "n_total_queries": len(rows),
        "n_errored": sum(1 for r in rows if r.get("error")),
        "overall": stage_block(rows),
        "overall_extras": extras_block(rows),
        "by_category": {},
        "by_category_extras": {},
        "unknown_predictions": {"n_predicted_unknown": sum(1 for r in rows if r.get("predicted_unknown"))},
        "avg_tokens_per_query": _nanmean([r.get("tokens_total") for r in rows]),
    }
    for cat in sorted({r.get("category", "") for r in rows}):
        sub = [r for r in rows if r.get("category", "") == cat]
        summary["by_category"][cat] = stage_block(sub)
        summary["by_category_extras"][cat] = extras_block(sub)
    return summary


def _fmt(v: Optional[float]) -> str:
    return "  n/a " if v is None else f"{v:6.3f}"


def print_summary(
    summary: Dict[str, Any],
    stages: List[Tuple[str, str, str]],
    stage_k: Optional[Dict[str, str]] = None,
) -> str:
    stage_k = stage_k or {}
    lines = ["=" * 100, "RETRIEVAL EVALUATION SUMMARY", "=" * 100,
             f"Total queries : {summary['n_total_queries']}   Errored: {summary['n_errored']}", ""]

    def block_table(title: str, block: Dict[str, Any], extras: Dict[str, Any]):
        lines.append(f"-- {title} " + "-" * max(0, 80 - len(title)))
        lines.append(
            f"{'stage':<22}{'k':>8}{'precision':>10}{'recall':>10}{'f1':>10}{'hit_rate':>10}"
            f"{'ctx_prec':>10}{'mrr':>10}{'avg_pred':>10}{'n':>6}"
        )
        for _, label, _shape in stages:
            s = block[label]
            lines.append(
                f"{label:<22}{stage_k.get(label, '-'):>8}"
                f"{_fmt(s['precision']):>10}{_fmt(s['recall']):>10}"
                f"{_fmt(s['f1']):>10}{_fmt(s['hit_rate']):>10}{_fmt(s['context_precision']):>10}"
                f"{_fmt(s['mrr']):>10}{_fmt(s['avg_n_pred']):>10}{s['n_queries']:>6}"
            )
        lines.append(f"Answer cosine (expected vs final): {_fmt(extras['answer_cosine']).strip()}")
        lines.append(f"Avg elapsed/query (s): {_fmt(extras['avg_elapsed_sec']).strip()}")
        if extras["timing_by_step"]:
            lines.append("Avg time by step (s/query, summed over sub-queries):")
            for step, v in extras["timing_by_step"].items():
                lines.append(f"  {step:<22}{_fmt(v)}")
        lines.append("")

    block_table("OVERALL", summary["overall"], summary["overall_extras"])
    for cat, block in summary["by_category"].items():
        block_table(f"CATEGORY: {cat}", block, summary["by_category_extras"][cat])

    uk = summary["unknown_predictions"]
    lines.append("-- 'UNKNOWN' PREDICTIONS " + "-" * 50)
    lines.append(f"Queries where final answer/confidence = Unknown/not_found: "
                 f"{uk['n_predicted_unknown']} / {summary['n_total_queries']}")
    lines.append(f"Average tokens/query: {summary.get('avg_tokens_per_query')}")
    lines.append("=" * 100)

    text = "\n".join(lines)
    print(text)
    return text


def run_evaluation(
    settings: Settings,
    index: ManualIndex,
    csv_path: str,
    output_prefix: str,
    retriever_name: str = "tree",
    limit: Optional[int] = None,
    start_at: int = 0,
    split_subqueries: bool = True,
    retriever: Optional[Retriever] = None,
) -> None:
    stages = get_stages(retriever_name, settings)
    timing_steps = get_timing_steps(retriever_name, settings)
    stage_k = get_stage_k(retriever_name, settings)
    fields = _build_csv_fields(retriever_name, settings)
    detailed_path = f"{output_prefix}_detailed.csv"

    # Built once (normally by the CLI) and reused for every row.
    if retriever is None:
        raise ValueError(
            "run_evaluation needs a prebuilt retriever (build it with the CLI's "
            "_make_retriever so the right deps are loaded)."
        )

    # Chunk-based retrievers: reuse the retriever's own lookup so the
    # evaluator and the retriever are guaranteed to see the same chunks.
    id_to_chunk: Optional[Dict[str, Dict[str, Any]]] = None
    if retriever_name in _CHUNK_BASED_RETRIEVERS:
        id_to_chunk = retriever.id_to_chunk

    # Embedder used for cosine(expected_answer, final_answer)
    embedder = getattr(retriever, "embedder", None)
    if embedder is None:
        logger.warning("No embedder on retriever; answer_cosine will be NaN.")

    df = pd.read_csv(csv_path)
    required = {"id", "category", "question", "expected_answer", "sections"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if start_at:
        df = df.iloc[start_at:]
    if limit:
        df = df.iloc[:limit]

    writer = IncrementalCsvWriter(detailed_path, fields, append=bool(start_at))

    try:
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            logger.info(f"({i}/{len(df)}) {row.get('id')} :: {str(row.get('question', ''))[:80]}...")
            t0 = time.time()
            r = evaluate_query(
                index, settings, row.to_dict(), retriever_name,
                split_subqueries=split_subqueries,
                id_to_chunk=id_to_chunk,
                retriever=retriever,
                embedder=embedder,
            )
            r["elapsed_sec"] = round(time.time() - t0, 2)
            writer.write_row(r)
    finally:
        writer.close()

    # Aggregate over the entire detailed CSV on disk, not just the rows
    # processed in this call, so a resumed run (start_at > 0) still
    # produces a summary covering every row written so far.
    rows_out = _load_detailed_csv(detailed_path)
    summary = aggregate(rows_out, stages, timing_steps)
    with open(f"{output_prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    summary_text = print_summary(summary, stages, stage_k)
    with open(f"{output_prefix}_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")