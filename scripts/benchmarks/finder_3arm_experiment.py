#!/usr/bin/env python3
"""FinDER ontology-size sweep: non-ontology → small → medium → large.

Measures how FIBO ontology size affects knowledge graph extraction quality
across S1-S6 slices. Ontology is the ONLY moving variable — same LLM, same
data, same gold metric. (CLAUDE.md §19 experiment design + §20 ethics apply.)

Arms (bench_common.ONTOLOGY_ARMS, nested supersets small ⊂ medium ⊂ large):
  non-ontology   []                                    → Entity/RELATED_TO floor
  small          be, ind                               → financial core only
  medium         be, ind, fbc, dbt, acc                → Goldilocks candidate
  large          be, ind, fbc, dbt, acc, fnd, sec, mkt, corp  → all 9 modules

Key hypotheses (CLAUDE.md §19):
  non-ontology < small  on S1/S2  ← ind adds typed FinancialMetric nodes
  small        < medium on S3-S5  ← fbc/dbt/acc fill missing slice coverage
  medium       > large  on S3-S5  ← fnd/sec/mkt/corp add noise, degrade quality
  S6_BASELINE_SINGLE: all arms ≈ neutral (single-passage control)

Opik trace contract (CLAUDE.md §19):
  Required  model:{spec}  dataset_index:{slice}/{case_id}
            prompt_hash:{10char}  ontology_hash:{10char-per-arm}
  Auxiliary phase:{arm}  variant:graph  slice  case  modules  meta_prompt  category
  name      {arm}/{case_id}/graph   e.g. medium/4af93b03/graph

Outputs:
  outputs/evaluation/finder_3arm_experiment/<run_prefix>/aggregate.json
  outputs/evaluation/finder_3arm_experiment/<run_prefix>/partial/<slice>_<case>_<arm>.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

from examples.finder.lib import bench_common as bc          # noqa: E402
from seocho.query.strategy import PromptTemplate             # noqa: E402
from finder_4arm_sample import (                             # noqa: E402
    load_sample, evaluate_answer, KGPromptTemplate,
    _ensure_db_ready,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_CSV = ROOT / "dataset" / "all_slices.csv"
_NUM_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*(?:%| million| billion| thousand)?", re.IGNORECASE)

# Canonical 129-case config lives in bench_common so ALL scripts share it.
FINAL_SLICE_CONFIG = bc.FINAL_SLICE_CONFIG

_ANSWER_SYSTEM = (
    "You are a financial analyst answering a question using ONLY the provided "
    "context (derived from SEC 10-K filings). Answer directly — no reasoning narration.\n"
    "- Ground every figure; preserve units, scale (thousands/millions), "
    "period (FY/quarter), and basis (GAAP/non-GAAP).\n"
    "- Show the arithmetic explicitly for any growth/ratio/delta.\n"
    "- If the needed figure is not in the context, say "
    "'not in the provided context' rather than guessing."
)

_INFRA_LABELS = {"Document", "DocumentVersion", "Chunk", "Section"}


def _verify_prompt_conditions(system_tmpl: str, user_tmpl: str) -> None:
    """Verify the 3 mandatory extraction prompt conditions and abort if any fail.

    1. {{ontology}} wired — KG schema injection point present in system prompt
    2. KG engineer role — explicit 'knowledge graph engineer' instruction
    3. {{text}} slot — raw source data placeholder present in user template
    """
    errors = []
    if "{{ontology}}" not in system_tmpl:
        errors.append("FAIL ① {{ontology}} not found in system prompt")
    if "knowledge graph engineer" not in system_tmpl.lower():
        errors.append("FAIL ② 'knowledge graph engineer' role not found in system prompt")
    if "{{text}}" not in user_tmpl:
        errors.append("FAIL ③ {{text}} not found in user template")
    if errors:
        for e in errors:
            print(f"  {e}", flush=True)
        raise SystemExit("Extraction prompt failed mandatory condition check. Aborting.")
    print("  ① {{ontology}} wired ✓", flush=True)
    print("  ② knowledge graph engineer role ✓", flush=True)
    print("  ③ {{text}} slot ✓", flush=True)


def _make_prompt_id() -> str:
    llm_spec = os.environ.get("SEOCHO_LLM", "grok/grok-4.3")
    provider = llm_spec.split("/")[0].lower() if "/" in llm_spec else llm_spec.lower()
    return f"{provider}_kg@v1"


PROMPT_ID = _make_prompt_id()


# ---------------------------------------------------------------------------
# Ontology helpers
# ---------------------------------------------------------------------------

def _ontology_hash(ontology) -> str:
    """Deterministic 10-char hash of an ontology's entity+relationship schema."""
    try:
        ctx = ontology.to_extraction_context()
        blob = ctx.get("entity_types", "") + "\n" + ctx.get("relationship_types", "")
    except Exception:
        try:
            import json
            blob = json.dumps(
                {"nodes": sorted(ontology.nodes.keys()),
                 "rels": sorted(ontology.relationships.keys())},
                sort_keys=True,
            )
        except Exception:
            blob = repr(ontology)
    return bc.short_hash(blob)


def _graph_context(graph_store, workspace_id: str, database: str) -> str:
    """Serialize domain nodes + relationships to text (graph-as-context).

    Infrastructure nodes (Document/Chunk/Section/DocumentVersion) and their
    relationships are excluded — they carry raw source text, not structured
    knowledge, and confuse the QA model into answering 'not in context'.
    """
    lines: list[str] = ["=== Knowledge graph: entities & metrics ==="]
    nodes = graph_store.query(
        "MATCH (n {_workspace_id:$w}) RETURN labels(n) AS l, properties(n) AS p",
        params={"w": workspace_id}, database=database,
    )
    domain_ids: set[str] = set()
    for r in nodes or []:
        labs = [x for x in (r["l"] or []) if x not in _INFRA_LABELS]
        if not labs:
            continue
        p = r["p"] or {}
        nm = p.get("name") or p.get("uri") or ""
        domain_ids.add(nm)
        bits = [f"{k}={p[k]}" for k in
                ("value", "period", "basis", "segment", "amount",
                 "coupon_rate", "maturity_date", "category", "standard")
                if p.get(k)]
        lines.append(f"- ({'/'.join(labs)}) {nm}" + (f" [{', '.join(bits)}]" if bits else ""))

    if len(domain_ids) == 0:
        return ""   # signal to caller: no domain content extracted

    # Only include relationships between domain nodes (skip infra rels).
    rels = graph_store.query(
        "MATCH (a {_workspace_id:$w})-[x]->(b {_workspace_id:$w}) "
        "WHERE NOT a:Document AND NOT a:DocumentVersion "
        "  AND NOT a:Chunk AND NOT a:Section "
        "RETURN coalesce(a.name,a.uri,'?') AS s, type(x) AS t, coalesce(b.name,b.uri,'?') AS o "
        "LIMIT 80",
        params={"w": workspace_id}, database=database,
    )
    if rels:
        lines.append("=== Relationships ===")
        for r in rels:
            lines.append(f"- {r['s']} -{r['t']}-> {r['o']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Single (case × arm) runner
# ---------------------------------------------------------------------------

def run_one(
    *,
    case: dict,
    arm_spec: bc.OntologyArmSpec,
    llm_spec: str,
    extraction_tmpl: PromptTemplate,
    prompt_hash: str,
    run_prefix: str,
    database: str,
    out_partial_dir: Path,
) -> list[dict]:
    """Run one (case × arm): extract graph then answer in graph-only mode.

    Returns a list with one result dict (graph mode).
    Hybrid can be computed offline from saved graph context + LanceDB embeddings.
    """
    from seocho import Seocho
    from seocho.store.graph import Neo4jGraphStore
    from seocho.store.llm import create_llm_backend
    from examples.finder.datasets.fibo_modules.compose import compose_modules

    arm = arm_spec.name
    modules = list(arm_spec.modules)
    ontology = compose_modules(modules)
    workspace_id = f"{run_prefix}-{arm}-{case['case_id']}"
    trace_name = bc.make_trace_name(arm, case["case_id"], "graph")
    modules_label = "+".join(modules) or "baseline"
    onto_hash = _ontology_hash(ontology)
    dataset_index = f"{case['slice']}/{case['case_id']}"
    provider, model = llm_spec.split("/", 1) if "/" in llm_spec else ("grok", llm_spec)
    meta_prompt_name = bc._resolve_meta_prompt_path().stem

    # All 4 required tags always present (CLAUDE.md §19 + bench_common fix).
    # prompt_hash and ontology_hash carry 10-char deterministic hex values.
    tags, metadata = bc.build_core_meta(
        dataset_name="all_slices.csv",
        dataset_index=dataset_index,
        case_id=case["case_id"],
        slice_tag=case["slice"],
        category=case["category"],
        llm_spec=llm_spec,
        provider=provider,
        mode="graph",
        reasoning_mode=False,
        repair_budget=0,
        flow="graphrag",
        ontology_hash=onto_hash,          # arm-specific 10-char hash
        ontology_modules=modules_label,
        prompt_hash=prompt_hash,           # extraction prompt hash
        run_prefix=run_prefix,
        workspace_id=workspace_id,
        extra_tags={
            "retrieval": "graph",
            "ontology": arm,
            # auxiliary (CLAUDE.md §19)
            "phase": arm,
            "variant": "graph",
            "case": case["case_id"],
            "modules": modules_label,
            "meta_prompt": meta_prompt_name,
            "category": case["category"],
            "slice": case["slice"],
        },
        extra_metadata={
            "ontology_arm": arm,
            "ontology_modules_list": modules,
            "ontology_node_count": len(ontology.nodes),
            "ontology_rel_count": len(ontology.relationships),
            "ontology_rationale": arm_spec.rationale,
            "slice_hypothesis": arm_spec.slice_hypothesis.get(case["slice"], "unknown"),
            "prompt_id": PROMPT_ID,
            "experiment_database": database,
            "case_query": case["query"],
            "case_n_refs": case["n_refs"],
            "case_type": case["type"],
        },
    )

    print(
        f"    {trace_name}: modules={modules_label} onto={onto_hash} "
        f"hyp={arm_spec.slice_hypothesis.get(case['slice'], '?')}",
        flush=True,
    )

    started = time.perf_counter()
    error = ""
    nodes_created = rels_created = 0
    add_ms = 0.0
    graph_ctx = ""
    llm = None
    client = None

    try:
        graph_store = Neo4jGraphStore(
            os.environ["NEO4J_URI"],
            os.environ.get("NEO4J_USER", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", ""),
        )
        llm = bc.create_experiment_llm(
            primary_spec=llm_spec,
            fallback_spec="openai/gpt-4o-mini",
        )
        client = Seocho(
            ontology=ontology,
            graph_store=graph_store,
            llm=llm,
            workspace_id=workspace_id,
            extraction_prompt=extraction_tmpl,
        )
        client.default_database = database
        try:
            graph_store.ensure_constraints(ontology, database=database)
        except Exception:
            pass

        t0 = time.perf_counter()
        for i, ref in enumerate(case["references"], 1):
            print(
                f"    {trace_name}: add ref {i}/{len(case['references'])} ({len(ref)} chars)",
                flush=True,
            )
            client.add(ref, user_id=workspace_id)
        add_ms = round((time.perf_counter() - t0) * 1000, 2)

        try:
            n_rows = graph_store.query(
                "MATCH (n {_workspace_id:$w}) RETURN count(n) AS c",
                params={"w": workspace_id}, database=database,
            )
            r_rows = graph_store.query(
                "MATCH (a {_workspace_id:$w})-[x]->() RETURN count(x) AS c",
                params={"w": workspace_id}, database=database,
            )
            nodes_created = int(n_rows[0]["c"]) if n_rows else 0
            rels_created = int(r_rows[0]["c"]) if r_rows else 0
        except Exception:
            pass

        graph_ctx = _graph_context(graph_store, workspace_id, database)
        if graph_ctx.strip():
            preview_lines = graph_ctx.splitlines()
            print(f"    {trace_name}: graph ({nodes_created}n/{rels_created}r):", flush=True)
            for line in preview_lines[:20]:
                print(f"      {line}", flush=True)
            if len(preview_lines) > 20:
                print(f"      ... ({len(preview_lines) - 20} more lines)", flush=True)
        else:
            print(f"    {trace_name}: graph empty → fallback to source text", flush=True)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    # Fallback LLM for QA if extraction-phase LLM creation failed.
    if llm is None:
        try:
            llm = bc.create_experiment_llm(
                primary_spec=llm_spec,
                fallback_spec="openai/gpt-4o-mini",
            )
        except Exception:
            pass

    # Build QA context: use graph when domain nodes were extracted,
    # fall back to raw source text when extraction produced nothing useful.
    # This ensures we measure "does graph structure add value?" not
    # "does extraction always succeed?" — both get a fair answer attempt.
    if graph_ctx.strip():
        qa_context = f"=== GRAPH CONTEXT ===\n{graph_ctx}"
        retrieval_source = "graph"
    else:
        # Extraction failed: fall back to raw source references (same provenance).
        qa_context = "=== SOURCE TEXT (graph extraction: no domain entities) ===\n" + \
                     "\n\n---\n\n".join(case["references"])
        retrieval_source = "graph_fallback_text"

    mode_specs = [
        ("graph", "graphrag", retrieval_source, qa_context),
    ]

    results: list[dict] = []
    meta_prompt_name = bc._resolve_meta_prompt_path().stem

    for mode_name, flow, retrieval_tag, qa_context in mode_specs:
        tname = bc.make_trace_name(arm, case["case_id"], mode_name)
        m_tags, m_meta = bc.build_core_meta(
            dataset_name="all_slices.csv",
            dataset_index=dataset_index,
            case_id=case["case_id"],
            slice_tag=case["slice"],
            category=case["category"],
            llm_spec=llm_spec,
            provider=provider,
            mode=mode_name,
            reasoning_mode=False,
            repair_budget=0,
            flow=flow,
            ontology_hash=onto_hash,
            ontology_modules=modules_label,
            prompt_hash=prompt_hash,
            run_prefix=run_prefix,
            workspace_id=workspace_id,
            extra_tags={
                "retrieval": retrieval_tag,
                "ontology": arm,
                "phase": arm,
                "variant": mode_name,
                "case": case["case_id"],
                "modules": modules_label,
                "meta_prompt": meta_prompt_name,
                "category": case["category"],
                "slice": case["slice"],
            },
            extra_metadata={
                "ontology_arm": arm,
                "ontology_modules_list": modules,
                "ontology_node_count": len(ontology.nodes),
                "ontology_rel_count": len(ontology.relationships),
                "slice_hypothesis": arm_spec.slice_hypothesis.get(case["slice"], "unknown"),
                "prompt_id": PROMPT_ID,
                "experiment_database": database,
                "case_query": case["query"],
                "case_n_refs": case["n_refs"],
                "nodes_created": nodes_created,
                "relationships_created": rels_created,
            },
        )

        ans_err = ""
        t1 = time.perf_counter()

        def _work(_ctx=qa_context, _exp=case["expected_answer"],
                  _tname=tname, _tags=m_tags, _meta=m_meta):
            _ans = ("not in the provided context" if not _ctx.strip()
                    else "LLM unavailable" if llm is None
                    else (lambda r: getattr(r, "text", None) or
                          getattr(r, "content", None) or str(r))(
                        llm.complete(
                            system=_ANSWER_SYSTEM,
                            user=f"Question: {case['query']}\n\n{_ctx}",
                        )
                    ))
            m = evaluate_answer(_exp, _ans)
            bc.set_opik_feedback_scores({
                "number_overlap": m["number_overlap_ratio"],
                "contains_match": 1.0 if m["contains_match"] else 0.0,
            })
            bc.set_opik_trace_metadata(name=_tname, tags=_tags, metadata=_meta)
            return _ans

        try:
            answer = bc.run_under_opik_track(
                name=tname, tags=m_tags, metadata=m_meta, work_fn=_work,
            )
        except Exception as exc:
            answer = ""
            ans_err = f"{type(exc).__name__}: {exc}"
        ask_ms = round((time.perf_counter() - t1) * 1000, 2)

        metrics = evaluate_answer(case["expected_answer"], answer)
        result = {
            "case_id": case["case_id"],
            "slice": case["slice"],
            "category": case["category"],
            "type": case["type"],
            "n_refs": case["n_refs"],
            "arm": arm,
            "mode": mode_name,
            "retrieval": retrieval_tag,
            "ontology_modules": modules,
            "ontology_hash": onto_hash,
            "ontology_node_count": len(ontology.nodes),
            "ontology_rel_count": len(ontology.relationships),
            "slice_hypothesis": arm_spec.slice_hypothesis.get(case["slice"], "unknown"),
            "model": llm_spec,
            "prompt_id": PROMPT_ID,
            "prompt_hash": prompt_hash,
            "workspace_id": workspace_id,
            "graph_backend": "neo4j",
            "database": database,
            "query": case["query"],
            "expected_answer": case["expected_answer"],
            "answer": answer,
            "evaluation": metrics,
            "latency_ms": {
                "add": add_ms,
                "ask": ask_ms,
                "total": round((time.perf_counter() - started) * 1000, 2),
            },
            "nodes_created": nodes_created,
            "relationships_created": rels_created,
            "graph_context_chars": len(graph_ctx),
            "error": error or ans_err,
        }
        try:
            bc.atomic_write_json(
                out_partial_dir / f"{case['slice']}_{case['case_id']}_{arm}_{mode_name}.json",
                result,
            )
        except Exception as exc:
            print(f"  [warn] partial write failed: {exc}", flush=True)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarize(results: list[dict]) -> dict:
    # key: (slice, arm, mode)
    by: dict[tuple[str, str, str], list[dict]] = {}
    for r in results:
        by.setdefault((r["slice"], r["arm"], r.get("mode", "graph")), []).append(r)
    out: dict[str, dict] = {}
    for (slc, arm, mode), runs in sorted(by.items()):
        ov = [r["evaluation"]["number_overlap_ratio"] for r in runs]
        ct = [r["evaluation"]["contains_match"] for r in runs]
        nodes = [r["nodes_created"] for r in runs]
        rels = [r["relationships_created"] for r in runs]
        hyp = runs[0].get("slice_hypothesis", "?") if runs else "?"
        out[f"{slc}|{arm}|{mode}"] = {
            "slice": slc,
            "arm": arm,
            "mode": mode,
            "slice_hypothesis": hyp,
            "n": len(runs),
            "number_overlap_mean": round(sum(ov) / len(ov), 3) if ov else 0.0,
            "contains_rate": round(sum(ct) / len(ct), 3) if ct else 0.0,
            "nodes_mean": round(sum(nodes) / len(nodes), 1) if nodes else 0.0,
            "rels_mean": round(sum(rels) / len(rels), 1) if rels else 0.0,
            "errors": sum(1 for r in runs if r["error"]),
        }
    return out


def print_table(summary: dict) -> None:
    print(
        "\nslice                   | arm           | hypothesis |  n "
        "| overlap | contains | nodes | rels  | err"
    )
    print("-" * 110)
    prev_slice = ""
    for row in summary.values():
        slc = row["slice"]
        sep = "·" if slc == prev_slice else slc[:23]
        prev_slice = slc
        hyp = row.get("slice_hypothesis", "?")[:7]
        print(
            f"{sep:<23} | {row['arm']:<13} | {hyp:<10} | {row['n']:2d} | "
            f"{row['number_overlap_mean']:.3f}   | {row['contains_rate']:.2f}     | "
            f"{row['nodes_mean']:5.1f} | {row['rels_mean']:5.1f} | {row['errors']}"
        )


def _print_arm_delta(summary: dict) -> None:
    """Print pairwise overlap deltas (medium−small, medium−large) per slice."""
    slices = sorted({v["slice"] for v in summary.values()})

    def _delta_row(slc: str, arm_a: str, arm_b: str) -> str:
        a = summary.get(f"{slc}|{arm_a}")
        b = summary.get(f"{slc}|{arm_b}")
        if a is None or b is None:
            return f"  {arm_b} n/a"
        delta = b["number_overlap_mean"] - a["number_overlap_mean"]
        hyp_b = b.get("slice_hypothesis", "?")
        verdict = (
            f"{arm_b} > {arm_a} ✓" if delta > 0.01 and hyp_b in {"graph"}
            else f"≈ tie" if abs(delta) <= 0.01
            else f"{arm_a} > {arm_b} (!)" if delta < -0.01 and hyp_b in {"graph"}
            else f"{arm_b} > {arm_a}" if delta > 0.01
            else f"{arm_a} > {arm_b}"
        )
        return f"{delta:>+7.3f}  {verdict}"

    print("\n=== Ontology size sweep: overlap delta per slice ===")
    print(f"{'slice':<25} | {'non→small':>17} | {'small→medium':>17} | {'medium→large':>17} | hyp(medium)")
    print("-" * 100)
    for slc in slices:
        hyp = (summary.get(f"{slc}|medium") or {}).get("slice_hypothesis", "?")
        c1 = _delta_row(slc, "non-ontology", "small")
        c2 = _delta_row(slc, "small", "medium")
        c3 = _delta_row(slc, "medium", "large")
        print(f"{slc:<25} | {c1:>17} | {c2:>17} | {c3:>17} | {hyp}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="FinDER 3-arm ontology experiment")
    ap.add_argument("--n-per-slice", type=int, default=20,
                    help="Default cases per slice. S4/S6 use FINAL_SLICE_CONFIG unless --ignore-final-config.")
    ap.add_argument("--ignore-final-config", action="store_true",
                    help="Use --n-per-slice uniformly (overrides FINAL_SLICE_CONFIG).")
    ap.add_argument("--graph-favorable", action="store_true",
                    help="Only sample graph-favorable cases: n_refs>=2, ref_len>=1500 "
                         "(S1 ref_len>=2000). Excludes S3_CO_COMP. "
                         "Uses GRAPH_FAVORABLE_CONFIG per-slice caps.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--llm", default=os.environ.get("SEOCHO_LLM", "grok/grok-4.3"))
    ap.add_argument(
        "--database",
        default=os.environ.get("SEOCHO_EXPERIMENT_DB", "finder3arm"),
        help="Fixed Neo4j experiment DB (sanitized name, no hyphens).",
    )
    ap.add_argument(
        "--arms",
        default="non-ontology,small,medium,large",
        help="Comma-separated arm names (non-ontology, small, medium, large).",
    )
    ap.add_argument("--limit-cases", type=int, default=0, help="Cap total cases (smoke). 0=all.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--run-prefix",
        default=f"3arm-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
    )
    args = ap.parse_args()

    bc.bootstrap(verbose=True)
    bc.set_global_determinism(args.seed)

    user_tmpl = "Source 10-K text to extract into the graph:\n\n{{text}}"
    system_tmpl = bc.load_meta_prompt()
    prompt_hash = bc.short_hash(system_tmpl)

    # Verify 3 mandatory extraction prompt conditions before any LLM/DB work.
    print("== verifying extraction prompt conditions ==")
    _verify_prompt_conditions(system_tmpl, user_tmpl)

    extraction_tmpl = KGPromptTemplate(system=system_tmpl, user=user_tmpl)
    resolved_prompt = bc._resolve_meta_prompt_path().name
    print(f"== extraction prompt: {resolved_prompt} ({len(system_tmpl)} chars hash={prompt_hash}) ==")

    requested_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    arm_specs = [bc.get_ontology_arm(n) for n in requested_names]

    if args.graph_favorable:
        # Graph-favorable: only cases with enough content for successful extraction.
        # Conditions derived from empirical analysis of extraction failure patterns.
        cases = load_sample(
            args.n_per_slice,
            args.seed,
            slice_overrides=bc.GRAPH_FAVORABLE_CONFIG,
            min_ref_len={"S1_FIN_COMP": 2000},
            min_n_refs={"S2_FIN_NONQUANT_MULTI": 2, "S4_CO_MULTI_NONQUANT": 2, "S5_FN_MULTI": 2},
            exclude_slices={"S3_CO_COMP"},
        )
    else:
        slice_cfg = None if args.ignore_final_config else FINAL_SLICE_CONFIG
        cases = load_sample(args.n_per_slice, args.seed, slice_overrides=slice_cfg)
    if args.limit_cases:
        cases = cases[: args.limit_cases]

    total = len(cases) * len(arm_specs)
    print(f"== plan: {len(cases)} cases × {len(arm_specs)} arms = {total} runs ==")
    for spec in arm_specs:
        mods = "+".join(spec.modules) or "baseline"
        print(f"   [{spec.name}] modules={mods}")
        print(f"     {spec.rationale[:110]}")
    by_slice: dict[str, int] = {}
    for c in cases:
        by_slice[c["slice"]] = by_slice.get(c["slice"], 0) + 1
    print(f"   sample per slice: {by_slice}")

    if args.dry_run:
        for c in cases:
            print(f"   {c['slice']:<23} {c['case_id']}  {c['query'][:70]}")
        print("(dry-run: stopping before LLM/graph work)")
        return 0

    # Experiment-traces-only: do NOT enable SEOCHO's OpikBackend (emits
    # sdk.extraction/sdk.query noise + wraps LLM via track_openai).
    # Traces come solely from bc.run_under_opik_track + set_opik_trace_metadata.
    print(
        f"== tracing: experiment-traces-only "
        f"project={os.environ.get('OPIK_PROJECT_NAME')} "
        f"ws={os.environ.get('OPIK_WORKSPACE')} =="
    )

    def flush_tracing() -> None:
        try:
            import opik
            opik.flush_tracker()
        except Exception:
            pass

    from seocho.store.graph import Neo4jGraphStore, sanitize_database_name
    database = sanitize_database_name(args.database)
    _gs = Neo4jGraphStore(
        os.environ["NEO4J_URI"],
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", ""),
    )
    _ensure_db_ready(_gs, database)
    _gs.close()
    print(f"== experiment database: {database} (online) ==")

    out_dir = ROOT / "outputs" / "evaluation" / "finder_3arm_experiment" / args.run_prefix
    out_partial = out_dir / "partial"
    out_partial.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    wall_start = time.perf_counter()
    run_i = 0
    for case in cases:
        for arm_spec in arm_specs:
            run_i += 1
            print(f"\n>>> [{run_i}/{total}] {case['slice']} {case['case_id']} ({arm_spec.name})")
            # Skip if already done (resume support).
            done_path = out_partial / f"{case['slice']}_{case['case_id']}_{arm_spec.name}_graph.json"
            if done_path.exists():
                try:
                    mode_results = [json.loads(done_path.read_text())]
                    print(f"  [skip] {bc.make_trace_name(arm_spec.name, case['case_id'], 'graph')} already done")
                    for res in mode_results:
                        results.append(res)
                    continue
                except Exception:
                    pass

            mode_results = run_one(
                case=case,
                arm_spec=arm_spec,
                llm_spec=args.llm,
                extraction_tmpl=extraction_tmpl,
                prompt_hash=prompt_hash,
                run_prefix=args.run_prefix,
                database=database,
                out_partial_dir=out_partial,
            )
            for res in mode_results:
                ev = res["evaluation"]
                mark = "OK" if not res["error"] else "ERR"
                print(
                    f"    [{res['mode']:<12}] {mark}  overlap={ev['number_overlap_ratio']:.2f} "
                    f"nums={ev['shared_numbers']}/{ev['expected_number_count']} "
                    f"contains={ev['contains_match']} "
                    f"nodes={res['nodes_created']} add={res['latency_ms']['add']}ms "
                    f"ask={res['latency_ms']['ask']}ms"
                )
                if res["error"]:
                    print(f"    error: {res['error']}")
                results.append(res)

    try:
        flush_tracing()
    except Exception:
        pass

    summary = summarize(results)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "finder_3arm",
        "run_prefix": args.run_prefix,
        "llm": args.llm,
        "seed": args.seed,
        "database": database,
        "n_per_slice": args.n_per_slice,
        "arms": [
            {"name": s.name, "modules": list(s.modules), "rationale": s.rationale,
             "slice_hypothesis": s.slice_hypothesis}
            for s in arm_specs
        ],
        "prompt_id": PROMPT_ID,
        "prompt_hash": prompt_hash,
        "opik_project": os.environ.get("OPIK_PROJECT_NAME", ""),
        "opik_workspace": os.environ.get("OPIK_WORKSPACE", ""),
        "total_runs": len(results),
        "total_wall_seconds": round(time.perf_counter() - wall_start, 2),
        "summary": summary,
        "results": results,
    }
    agg = out_dir / "aggregate.json"
    bc.atomic_write_json(agg, payload)
    print(f"\n== wrote {agg.relative_to(ROOT)} ==")
    print_table(summary)
    _print_arm_delta(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
