#!/usr/bin/env python3
"""Quick vector vs graph vs hybrid comparison on hand-picked graph-favorable cases.

Runs medium arm (Goldilocks ontology) in graph + hybrid modes on 12 cases
pre-selected from S2/S4/S5 (n_refs>=2, highest ref_len) where graph is
theoretically strongest. Vector scores are already in vector-v2 results.

Total: 12 cases × 1 arm × 2 modes = 24 runs  (~30 min)
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

from examples.finder.lib import bench_common as bc
from finder_4arm_sample import evaluate_answer, KGPromptTemplate, _ensure_db_ready
from finder_3arm_experiment import (
    _ontology_hash, _graph_context, _verify_prompt_conditions,
    _ANSWER_SYSTEM, _INFRA_LABELS, PROMPT_ID,
)
from finder_4arm_sample import _vector_context as _vc_fn

# Hand-picked: S2/S4/S5, top-4 ref_len, already in vector-v2
TARGET_CASE_IDS = [
    # S2 (IS+BS cross-statement)
    "4f87bbef", "347bdeaa", "92a58726", "dc4dc72e",
    # S4 (cross-segment, long docs)
    "c4c47e11", "07a20cbe", "9a55a405", "38e0b21f",
    # S5 (footnote integration)
    "05095fe2", "7cc7ebab", "6b611c81", "e55b23c3",
]


def load_target_cases() -> list[dict]:
    import pandas as pd
    REF_SEP = "===EVIDENCE_BOUNDARY==="
    df = pd.read_csv(ROOT / "dataset" / "all_slices.csv")
    rows = df[df["_id"].isin(TARGET_CASE_IDS)]
    cases = []
    for _, r in rows.iterrows():
        refs = [x.strip() for x in str(r["references_joined"]).split(REF_SEP) if x.strip()]
        cases.append({
            "case_id": r["_id"], "slice": r["slice"],
            "category": r["category"],
            "type": r["type"] if isinstance(r["type"], str) else "",
            "n_refs": int(r["n_refs"]),
            "query": r["query"],
            "expected_answer": r["answer"],
            "references": refs,
        })
    # preserve TARGET_CASE_IDS order
    order = {cid: i for i, cid in enumerate(TARGET_CASE_IDS)}
    return sorted(cases, key=lambda c: order.get(c["case_id"], 99))


def run_one(*, case, arm_spec, llm_spec, extraction_tmpl, prompt_hash,
            run_prefix, database, oai_client, out_dir) -> list[dict]:
    from seocho import Seocho
    from seocho.store.graph import Neo4jGraphStore
    from examples.finder.datasets.fibo_modules.compose import compose_modules
    arm = arm_spec.name
    modules = list(arm_spec.modules)
    ontology = compose_modules(modules)
    workspace_id = f"{run_prefix}-{arm}-{case['case_id']}"
    onto_hash = _ontology_hash(ontology)
    modules_label = "+".join(modules) or "baseline"
    dataset_index = f"{case['slice']}/{case['case_id']}"
    provider, model = llm_spec.split("/", 1) if "/" in llm_spec else ("grok", llm_spec)
    meta_prompt_name = bc._resolve_meta_prompt_path().stem

    started = time.perf_counter()
    error = ""
    nodes_created = rels_created = 0
    add_ms = 0.0
    graph_ctx = ""
    llm = None

    try:
        gs = Neo4jGraphStore(os.environ["NEO4J_URI"],
                             os.environ.get("NEO4J_USER", "neo4j"),
                             os.environ.get("NEO4J_PASSWORD", ""))
        llm = bc.create_experiment_llm(llm_spec, "openai/gpt-4o-mini")
        client = Seocho(ontology=ontology, graph_store=gs, llm=llm,
                        workspace_id=workspace_id, extraction_prompt=extraction_tmpl)
        client.default_database = database
        try:
            gs.ensure_constraints(ontology, database=database)
        except Exception:
            pass

        t0 = time.perf_counter()
        for i, ref in enumerate(case["references"], 1):
            print(f"    add ref {i}/{len(case['references'])} ({len(ref)} chars)", flush=True)
            client.add(ref, user_id=workspace_id)
        add_ms = round((time.perf_counter() - t0) * 1000, 2)

        try:
            n = gs.query("MATCH (n {_workspace_id:$w}) RETURN count(n) AS c",
                         params={"w": workspace_id}, database=database)
            r = gs.query("MATCH (a {_workspace_id:$w})-[x]->() RETURN count(x) AS c",
                         params={"w": workspace_id}, database=database)
            nodes_created = int(n[0]["c"]) if n else 0
            rels_created  = int(r[0]["c"]) if r else 0
        except Exception:
            pass

        graph_ctx = _graph_context(gs, workspace_id, database)
        status = f"graph ({nodes_created}n/{rels_created}r)" if graph_ctx.strip() else "graph empty → text fallback"
        print(f"    {status}", flush=True)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        try:
            client.close()  # type: ignore
        except Exception:
            pass

    if llm is None:
        try:
            llm = bc.create_experiment_llm(llm_spec, "openai/gpt-4o-mini")
        except Exception:
            pass

    # Build contexts
    if graph_ctx.strip():
        graph_qa_ctx = f"=== GRAPH CONTEXT ===\n{graph_ctx}"
        retrieval_src = "graph"
    else:
        graph_qa_ctx = "=== SOURCE TEXT (graph extraction: no domain entities) ===\n" + \
                       "\n\n---\n\n".join(case["references"])
        retrieval_src = "graph_fallback_text"

    vec_ctx = ""
    vec_error = ""
    try:
        vec_ctx = _vc_fn(case["references"], case["query"], oai_client)
    except Exception as exc:
        vec_error = f"{type(exc).__name__}: {exc}"

    hybrid_ctx = ""
    if not vec_error and graph_ctx.strip():
        hybrid_ctx = (f"=== VECTOR CONTEXT ===\n{vec_ctx}"
                      f"\n\n=== GRAPH CONTEXT ===\n{graph_ctx}")

    mode_specs = [("graph", "graph", graph_qa_ctx, retrieval_src)]
    if hybrid_ctx:
        mode_specs.append(("hybrid", "vector_graph", hybrid_ctx, "vector_graph"))

    results = []
    for mode_name, flow, ctx, ret_tag in mode_specs:
        tname = bc.make_trace_name(arm, case["case_id"], mode_name)
        tags, metadata = bc.build_core_meta(
            dataset_name="all_slices.csv", dataset_index=dataset_index,
            case_id=case["case_id"], slice_tag=case["slice"], category=case["category"],
            llm_spec=llm_spec, provider=provider, mode=mode_name,
            reasoning_mode=False, repair_budget=0, flow=flow,
            ontology_hash=onto_hash, ontology_modules=modules_label,
            prompt_hash=prompt_hash, run_prefix=run_prefix, workspace_id=workspace_id,
            extra_tags={"retrieval": ret_tag, "ontology": arm, "phase": arm,
                        "variant": mode_name, "case": case["case_id"],
                        "modules": modules_label, "meta_prompt": meta_prompt_name,
                        "slice": case["slice"], "category": case["category"]},
        )

        answer = ""
        ans_err = ""
        t1 = time.perf_counter()

        def _work(_c=ctx, _e=case["expected_answer"], _n=tname, _t=tags, _m=metadata):
            _ans = ("not in the provided context" if not _c.strip() else
                    "LLM unavailable" if llm is None else
                    (lambda rr: getattr(rr, "text", None) or str(rr))(
                        llm.complete(system=_ANSWER_SYSTEM,
                                     user=f"Question: {case['query']}\n\n{_c}")))
            ev = evaluate_answer(_e, _ans)
            bc.set_opik_feedback_scores({"number_overlap": ev["number_overlap_ratio"],
                                         "contains_match": 1.0 if ev["contains_match"] else 0.0})
            bc.set_opik_trace_metadata(name=_n, tags=_t, metadata=_m)
            return _ans

        try:
            answer = bc.run_under_opik_track(name=tname, tags=tags, metadata=metadata, work_fn=_work)
        except Exception as exc:
            answer = ""; ans_err = f"{type(exc).__name__}: {exc}"
        ask_ms = round((time.perf_counter() - t1) * 1000, 2)

        metrics = evaluate_answer(case["expected_answer"], answer)
        res = {
            "case_id": case["case_id"], "slice": case["slice"],
            "category": case["category"], "n_refs": case["n_refs"],
            "arm": arm, "mode": mode_name, "retrieval": ret_tag,
            "ontology_modules": modules, "ontology_hash": onto_hash,
            "model": llm_spec, "prompt_hash": prompt_hash,
            "query": case["query"], "expected_answer": case["expected_answer"],
            "answer": answer, "evaluation": metrics,
            "latency_ms": {"add": add_ms, "ask": ask_ms,
                           "total": round((time.perf_counter() - started) * 1000, 2)},
            "nodes_created": nodes_created, "relationships_created": rels_created,
            "error": error or ans_err,
        }
        bc.atomic_write_json(out_dir / f"{case['slice']}_{case['case_id']}_{arm}_{mode_name}.json", res)
        results.append(res)
    return results


def main() -> int:
    import argparse
    from openai import OpenAI
    from seocho.store.graph import Neo4jGraphStore, sanitize_database_name

    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default=os.environ.get("SEOCHO_LLM", "grok/grok-4.3"))
    ap.add_argument("--database", default="finderquick")
    ap.add_argument("--run-prefix",
                    default=f"quick-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}")
    ap.add_argument("--arms", default="medium",
                    help="Comma-separated arms. Default: medium only.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    bc.bootstrap(verbose=True)

    user_tmpl = "Source 10-K text to extract into the graph:\n\n{{text}}"
    system_tmpl = bc.load_meta_prompt()
    prompt_hash = bc.short_hash(system_tmpl)
    print("== verifying extraction prompt conditions ==")
    _verify_prompt_conditions(system_tmpl, user_tmpl)
    extraction_tmpl = KGPromptTemplate(system=system_tmpl, user=user_tmpl)
    print(f"== prompt hash={prompt_hash} ==")

    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    arm_specs = [bc.get_ontology_arm(n) for n in arm_names]
    cases = load_target_cases()
    total = len(cases) * len(arm_specs)

    print(f"== quick compare: {len(cases)} cases × {len(arm_specs)} arms ({arm_names}) = {total} runs ==")
    for c in cases:
        print(f"   {c['slice']:<30} {c['case_id']}  n_refs={c['n_refs']}  q={c['query'][:60]}")

    if args.dry_run:
        print("(dry-run)")
        return 0

    database = sanitize_database_name(args.database)
    _gs = Neo4jGraphStore(os.environ["NEO4J_URI"], os.environ.get("NEO4J_USER", "neo4j"),
                          os.environ.get("NEO4J_PASSWORD", ""))
    _ensure_db_ready(_gs, database)
    _gs.close()
    print(f"== database: {database} ==")

    oai_client = OpenAI(timeout=60)
    out_dir = ROOT / "outputs" / "evaluation" / "finder_quick_compare" / args.run_prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    wall = time.perf_counter()
    run_i = 0
    for case in cases:
        for arm_spec in arm_specs:
            run_i += 1
            print(f"\n>>> [{run_i}/{total}] {case['slice']} {case['case_id']} ({arm_spec.name})")
            mode_results = run_one(
                case=case, arm_spec=arm_spec, llm_spec=args.llm,
                extraction_tmpl=extraction_tmpl, prompt_hash=prompt_hash,
                run_prefix=args.run_prefix, database=database,
                oai_client=oai_client, out_dir=out_dir,
            )
            for res in mode_results:
                ev = res["evaluation"]
                mark = "OK" if not res["error"] else "ERR"
                print(f"    [{res['mode']:<6}] {mark}  overlap={ev['number_overlap_ratio']:.2f} "
                      f"contains={ev['contains_match']}  nodes={res['nodes_created']} "
                      f"ask={res['latency_ms']['ask']}ms")
            all_results.extend(mode_results)

    # ── comparison table ──
    vec_p = ROOT / "outputs" / "evaluation" / "finder_vector_arm" / "vector-v2" / "partial"
    vec = {json.loads(f.read_text())["case_id"]: json.loads(f.read_text())
           for f in vec_p.glob("*.json")}

    print("\n" + "=" * 90)
    print("VECTOR vs GRAPH vs HYBRID  (number_overlap, same cases, same gold)")
    print("=" * 90)
    print(f"{'case_id':<12} {'slice':<26} {'n_refs':>6} | {'vector':>7} {'graph':>7} {'hybrid':>7} | winner")
    print("-" * 90)

    by_case: dict[str, dict] = {}
    for r in all_results:
        by_case.setdefault(r["case_id"], {})[r["mode"]] = r["evaluation"]["number_overlap_ratio"]

    for cid in TARGET_CASE_IDS:
        if cid not in by_case:
            continue
        v = vec.get(cid, {}).get("evaluation", {}).get("number_overlap_ratio", "-")
        g = by_case[cid].get("graph", "-")
        h = by_case[cid].get("hybrid", "-")
        slc = vec.get(cid, {}).get("slice", "?")
        n_refs = vec.get(cid, {}).get("n_refs", "?")
        v_f = f"{v:.2f}" if isinstance(v, float) else "-"
        g_f = f"{g:.2f}" if isinstance(g, float) else "-"
        h_f = f"{h:.2f}" if isinstance(h, float) else "-"
        scores = {k: s for k, s in [("vector", v), ("graph", g), ("hybrid", h)]
                  if isinstance(s, float)}
        winner = max(scores, key=scores.get) if scores else "?"
        print(f"{cid:<12} {slc:<26} {n_refs:>6} | {v_f:>7} {g_f:>7} {h_f:>7} | {winner}")

    # per-slice summary
    print()
    print(f"{'slice':<30} | {'vec mean':>8} {'graph':>8} {'hybrid':>8}")
    print("-" * 60)
    from collections import defaultdict
    import statistics
    by_slc: dict[str, dict] = defaultdict(lambda: {"v": [], "g": [], "h": []})
    for cid in TARGET_CASE_IDS:
        if cid not in by_case: continue
        slc = vec.get(cid, {}).get("slice", "?")
        v = vec.get(cid, {}).get("evaluation", {}).get("number_overlap_ratio")
        if v is not None: by_slc[slc]["v"].append(v)
        g = by_case[cid].get("graph")
        if g is not None: by_slc[slc]["g"].append(g)
        h = by_case[cid].get("hybrid")
        if h is not None: by_slc[slc]["h"].append(h)
    for slc, d in sorted(by_slc.items()):
        vm = f"{statistics.mean(d['v']):.3f}" if d["v"] else "-"
        gm = f"{statistics.mean(d['g']):.3f}" if d["g"] else "-"
        hm = f"{statistics.mean(d['h']):.3f}" if d["h"] else "-"
        print(f"{slc:<30} | {vm:>8} {gm:>8} {hm:>8}")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_prefix": args.run_prefix, "llm": args.llm, "arms": arm_names,
        "cases": TARGET_CASE_IDS, "total_wall_seconds": round(time.perf_counter() - wall, 2),
        "results": all_results,
    }
    bc.atomic_write_json(out_dir / "results.json", payload)
    print(f"\n== wrote {(out_dir / 'results.json').relative_to(ROOT)} ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
