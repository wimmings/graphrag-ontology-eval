#!/usr/bin/env python3
"""Graph serialization format experiment.

Re-uses the 12 graphs already extracted in finderquick DB (quick-v1, medium arm).
NO re-extraction — QA only. Compares 4 serialization formats:

  current  — "- (NetIncome) $5.23 [period=FY2024]"           (existing)
  table    — markdown table with Label/Name/Value/Period/Basis
  nl       — natural language sentences ("Fiserv's EPS in FY2024 was $5.23")
  cypher   — Cypher-style triples  (Fiserv)-[:HAS_METRIC]->(EPS {value:"$5.23"})

Opik tags (CLAUDE.md §19):
  Required: model:{spec}  dataset_index:{slice}/{case_id}
            prompt_hash:{10char}  ontology_hash:{10char}
  Auxiliary: format:{fmt}  slice  case  experiment:serialization

Neo4j queries (see Appendix at bottom of file):
  View all domain nodes   → MATCH (n {_workspace_id:$ws}) ...
  View relationships      → MATCH (a {_workspace_id:$ws})-[r]->(b) ...
  Compare format scores   → in judged output JSON

Total: 12 cases × 4 formats = 48 QA calls  (~15 min, ~$2)
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

from examples.finder.lib import bench_common as bc
from finder_4arm_sample import evaluate_answer

# ── 12 케이스 (quick-v1 medium arm, 이미 추출 완료) ──────────────────────
TARGET_CASES = [
    ("S2_FIN_NONQUANT_MULTI", "4f87bbef"),
    ("S2_FIN_NONQUANT_MULTI", "347bdeaa"),
    ("S2_FIN_NONQUANT_MULTI", "92a58726"),
    ("S2_FIN_NONQUANT_MULTI", "dc4dc72e"),
    ("S4_CO_MULTI_NONQUANT",  "c4c47e11"),
    ("S4_CO_MULTI_NONQUANT",  "07a20cbe"),
    ("S4_CO_MULTI_NONQUANT",  "9a55a405"),
    ("S4_CO_MULTI_NONQUANT",  "38e0b21f"),
    ("S5_FN_MULTI",           "05095fe2"),
    ("S5_FN_MULTI",           "7cc7ebab"),
    ("S5_FN_MULTI",           "6b611c81"),
    ("S5_FN_MULTI",           "e55b23c3"),
]
SOURCE_DB       = "finderquick"
SOURCE_PREFIX   = "quick-v1-medium"
INFRA = {"Document", "DocumentVersion", "Chunk", "Section"}

_ANSWER_SYSTEM = (
    "You are a financial analyst. Answer the question using ONLY the provided context "
    "(SEC 10-K filings). Answer directly — no reasoning narration.\n"
    "- Ground every figure; preserve units, scale, period, basis.\n"
    "- Show arithmetic explicitly for growth/ratio/delta.\n"
    "- If the figure is not in the context, say 'not in the provided context'."
)

ONTOLOGY_HASH = "799a63e842"  # medium arm hash (from quick-v1)


# ── 직렬화 함수 4가지 ────────────────────────────────────────────────────

def _domain_nodes(gs, ws: str) -> list[dict]:
    rows = gs.query(
        "MATCH (n {_workspace_id:$w}) RETURN labels(n) AS l, properties(n) AS p",
        params={"w": ws}, database=SOURCE_DB,
    )
    out = []
    for r in rows or []:
        labs = [x for x in (r["l"] or []) if x not in INFRA]
        if not labs:
            continue
        p = r["p"] or {}
        out.append({"labels": labs, "props": p})
    return out


def _domain_rels(gs, ws: str) -> list[dict]:
    rows = gs.query(
        "MATCH (a {_workspace_id:$w})-[x]->(b {_workspace_id:$w}) "
        "WHERE NOT a:Document AND NOT a:DocumentVersion "
        "  AND NOT a:Chunk AND NOT a:Section "
        "RETURN coalesce(a.name,a.uri,'?') AS s, type(x) AS t, "
        "       coalesce(b.name,b.uri,'?') AS o LIMIT 60",
        params={"w": ws}, database=SOURCE_DB,
    )
    return rows or []


def fmt_current(nodes, rels) -> str:
    lines = ["=== Knowledge graph ==="]
    for n in nodes:
        nm = n["props"].get("name") or n["props"].get("uri") or ""
        bits = [f"{k}={n['props'][k]}" for k in
                ("value", "period", "basis", "amount", "coupon_rate",
                 "maturity_date", "category", "standard") if n["props"].get(k)]
        lines.append(f"- ({'/'.join(n['labels'])}) {nm}" +
                     (f"  [{', '.join(bits)}]" if bits else ""))
    if rels:
        lines.append("--- Relationships ---")
        for r in rels:
            lines.append(f"  {r['s']} -[{r['t']}]-> {r['o']}")
    return "\n".join(lines)


def fmt_table(nodes, rels) -> str:
    lines = ["| Label | Name | Value | Period | Basis/Category |",
             "|-------|------|-------|--------|----------------|"]
    for n in nodes:
        p = n["props"]
        label = "/".join(n["labels"])
        name  = p.get("name") or p.get("uri") or "-"
        val   = p.get("value") or p.get("amount") or "-"
        period = p.get("period") or "-"
        basis  = p.get("basis") or p.get("category") or p.get("standard") or "-"
        lines.append(f"| {label} | {name} | {val} | {period} | {basis} |")
    if rels:
        lines.append("")
        lines.append("**Relationships:**")
        for r in rels:
            lines.append(f"- `{r['s']}` → `{r['t']}` → `{r['o']}`")
    return "\n".join(lines)


def fmt_nl(nodes, rels) -> str:
    """Natural language sentences per node."""
    sentences = []
    for n in nodes:
        p = n["props"]
        label = n["labels"][0]
        name  = p.get("name") or p.get("uri") or "Unknown entity"
        parts = [f"{name} is a {label}"]
        if p.get("value"):   parts.append(f"with value {p['value']}")
        if p.get("amount"):  parts.append(f"amount {p['amount']}")
        if p.get("period"):  parts.append(f"for period {p['period']}")
        if p.get("basis"):   parts.append(f"({p['basis']} basis)")
        sentences.append(". ".join(parts) + ".")
    if rels:
        sentences.append("")
        for r in rels:
            sentences.append(f"{r['s']} {r['t'].lower().replace('_',' ')} {r['o']}.")
    return "\n".join(sentences)


def fmt_cypher(nodes, rels) -> str:
    """Cypher-style triple notation."""
    lines = ["// Knowledge graph (Cypher notation)"]
    for n in nodes:
        label = ":".join(n["labels"])
        p = n["props"]
        props = {k: v for k, v in p.items()
                 if k in ("name","value","amount","period","basis","coupon_rate",
                          "maturity_date","standard","ticker") and v}
        prop_str = ", ".join(f'{k}: "{v}"' for k, v in props.items())
        lines.append(f"(:{label} {{{prop_str}}})")
    if rels:
        lines.append("")
        for r in rels:
            s, t, o = r["s"], r["t"], r["o"]
            lines.append(f'("{s}")-[:{t}]->("{o}")')
    return "\n".join(lines)


FORMATS = {
    "current": fmt_current,
    "table":   fmt_table,
    "nl":      fmt_nl,
    "cypher":  fmt_cypher,
}


# ── 케이스 데이터 로드 ───────────────────────────────────────────────────

def load_cases() -> dict[str, dict]:
    import pandas as pd
    REF_SEP = "===EVIDENCE_BOUNDARY==="
    df = pd.read_csv(ROOT / "dataset" / "all_slices.csv")
    case_ids = [cid for _, cid in TARGET_CASES]
    rows = df[df["_id"].isin(case_ids)]
    out = {}
    for _, r in rows.iterrows():
        refs = [x.strip() for x in str(r["references_joined"]).split(REF_SEP) if x.strip()]
        out[r["_id"]] = {
            "case_id": r["_id"], "slice": r["slice"],
            "category": r["category"], "n_refs": int(r["n_refs"]),
            "query": r["query"], "expected_answer": r["answer"],
            "references": refs,
        }
    return out


# ── 메인 ────────────────────────────────────────────────────────────────

def main() -> int:
    import argparse
    from seocho.store.graph import Neo4jGraphStore

    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default=os.environ.get("SEOCHO_LLM", "grok/grok-4.3"))
    ap.add_argument("--formats", default="current,table,nl,cypher")
    ap.add_argument("--run-prefix",
                    default=f"fmt-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    bc.bootstrap(verbose=True)

    fmts = [f.strip() for f in args.formats.split(",") if f.strip() in FORMATS]
    cases = load_cases()
    total = len(TARGET_CASES) * len(fmts)

    # QA system prompt hash → prompt_hash 태그
    prompt_hash = bc.short_hash(_ANSWER_SYSTEM)

    print(f"== serialization format experiment ==")
    print(f"   formats: {fmts}")
    print(f"   cases:   {len(TARGET_CASES)}")
    print(f"   total:   {total} QA calls  (~{total*5//60}m{total*5%60}s)")
    print(f"   prompt_hash: {prompt_hash}")
    print(f"   ontology_hash: {ONTOLOGY_HASH}  (medium arm, finderquick)")
    print()

    if args.dry_run:
        for slc, cid in TARGET_CASES:
            print(f"   {slc:<28} {cid}  n_refs={cases[cid]['n_refs']}")
        print("(dry-run)")
        return 0

    print(f"== tracing: project={os.environ.get('OPIK_PROJECT_NAME')} "
          f"ws={os.environ.get('OPIK_WORKSPACE')} ==")

    llm = bc.create_experiment_llm(args.llm, "openai/gpt-4o-mini")
    gs  = Neo4jGraphStore(os.environ["NEO4J_URI"],
                          os.environ.get("NEO4J_USER", "neo4j"),
                          os.environ.get("NEO4J_PASSWORD", ""))

    out_dir = ROOT / "outputs" / "evaluation" / "finder_format_experiment" / args.run_prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    wall = time.perf_counter()
    run_i = 0

    for slc, cid in TARGET_CASES:
        case = cases[cid]
        ws   = f"{SOURCE_PREFIX}-{cid}"
        dataset_index = f"{slc}/{cid}"

        # fetch graph once, serialize in all formats
        nodes = _domain_nodes(gs, ws)
        rels  = _domain_rels(gs, ws)
        print(f"\n--- {slc} {cid}  ({len(nodes)} domain nodes) ---")

        for fmt in fmts:
            run_i += 1
            ctx  = FORMATS[fmt](nodes, rels)
            tags, metadata = bc.build_core_meta(
                dataset_name="all_slices.csv",
                dataset_index=dataset_index,
                case_id=cid,
                slice_tag=slc,
                category=case["category"],
                llm_spec=args.llm,
                provider=args.llm.split("/")[0],
                mode=fmt,
                reasoning_mode=False,
                repair_budget=0,
                flow="graph_fmt",
                ontology_hash=ONTOLOGY_HASH,
                ontology_modules="be+ind+fbc+dbt+acc",
                prompt_hash=prompt_hash,
                run_prefix=args.run_prefix,
                extra_tags={
                    "format": fmt,           # 핵심 실험 변수
                    "experiment": "serialization",
                    "retrieval": "graph",
                    "phase": "format-sweep",
                    "variant": fmt,
                    "slice": slc,
                    "case": cid,
                },
                extra_metadata={
                    "domain_nodes": len(nodes),
                    "domain_rels": len(rels),
                    "ctx_chars": len(ctx),
                    "case_query": case["query"],
                    "n_refs": case["n_refs"],
                    "source_workspace": ws,
                    "source_db": SOURCE_DB,
                },
            )

            tname = bc.make_trace_name(fmt, cid, "graph")

            answer = ""
            ans_err = ""
            t0 = time.perf_counter()

            def _work(_ctx=ctx, _exp=case["expected_answer"],
                      _n=tname, _t=tags, _m=metadata):
                _ans = ("not in the provided context" if not _ctx.strip() else
                        (lambda r: getattr(r, "text", None) or str(r))(
                            llm.complete(
                                system=_ANSWER_SYSTEM,
                                user=f"Question: {case['query']}\n\n{_ctx}",
                            )))
                ev = evaluate_answer(_exp, _ans)
                bc.set_opik_feedback_scores({
                    "number_overlap": ev["number_overlap_ratio"],
                    "contains_match": 1.0 if ev["contains_match"] else 0.0,
                })
                bc.set_opik_trace_metadata(name=_n, tags=_t, metadata=_m)
                return _ans

            try:
                answer = bc.run_under_opik_track(
                    name=tname, tags=tags, metadata=metadata, work_fn=_work)
            except Exception as exc:
                ans_err = f"{type(exc).__name__}: {exc}"
            ask_ms = round((time.perf_counter() - t0) * 1000, 2)

            metrics = evaluate_answer(case["expected_answer"], answer)
            res = {
                "case_id": cid, "slice": slc, "n_refs": case["n_refs"],
                "format": fmt,
                "domain_nodes": len(nodes), "domain_rels": len(rels),
                "ctx_chars": len(ctx),
                "model": args.llm, "prompt_hash": prompt_hash,
                "ontology_hash": ONTOLOGY_HASH,
                "query": case["query"],
                "expected_answer": case["expected_answer"],
                "answer": answer,
                "evaluation": metrics,
                "latency_ms": ask_ms,
                "error": ans_err,
            }
            bc.atomic_write_json(out_dir / f"{slc}_{cid}_{fmt}.json", res)
            all_results.append(res)

            ov = metrics["number_overlap_ratio"]
            mark = "✓" if ov > 0 else "·"
            print(f"  [{run_i:>2}/{total}] {fmt:<8} overlap={ov:.2f}  {mark}  ask={ask_ms:.0f}ms")

    gs.close()

    try:
        import opik; opik.flush_tracker()
    except Exception:
        pass

    # ── summary table ──────────────────────────────────────────────────
    import statistics, collections
    by_fmt: dict[str, list[float]] = collections.defaultdict(list)
    by_slc_fmt: dict[tuple, list[float]] = collections.defaultdict(list)
    for r in all_results:
        ov = r["evaluation"]["number_overlap_ratio"]
        by_fmt[r["format"]].append(ov)
        by_slc_fmt[(r["slice"], r["format"])].append(ov)

    print("\n" + "=" * 70)
    print("FORMAT COMPARISON  (number_overlap — judge_score via finder_judge.py)")
    print("=" * 70)
    print(f"{'format':<10} {'overall':>8} {'S2':>6} {'S4':>6} {'S5':>6}  rank")
    print("-" * 55)
    ranked = sorted(by_fmt.items(), key=lambda x: -statistics.mean(x[1]))
    for i, (fmt, scores) in enumerate(ranked, 1):
        overall = statistics.mean(scores)
        s2 = statistics.mean(by_slc_fmt.get(("S2_FIN_NONQUANT_MULTI", fmt), [0]))
        s4 = statistics.mean(by_slc_fmt.get(("S4_CO_MULTI_NONQUANT",  fmt), [0]))
        s5 = statistics.mean(by_slc_fmt.get(("S5_FN_MULTI",           fmt), [0]))
        best = " ◀ BEST" if i == 1 else ""
        print(f"{fmt:<10} {overall:>8.3f} {s2:>6.3f} {s4:>6.3f} {s5:>6.3f}  #{i}{best}")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_prefix": args.run_prefix, "llm": args.llm, "formats": fmts,
        "total_wall_seconds": round(time.perf_counter() - wall, 2),
        "results": all_results,
    }
    bc.atomic_write_json(out_dir / "results.json", payload)
    print(f"\n== wrote {(out_dir / 'results.json').relative_to(ROOT)} ==")
    print()
    print("다음 단계: judge_score 채점")
    print(f"  python scripts/benchmarks/finder_judge.py \\")
    print(f"    --inputs 'outputs/evaluation/finder_format_experiment/{args.run_prefix}/S*.json' \\")
    print(f"    --out outputs/evaluation/judged-format-{args.run_prefix}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ═══════════════════════════════════════════════════════════════════════
# APPENDIX: Opik & Neo4j 보기
# ═══════════════════════════════════════════════════════════════════════
#
# ── Opik UI 필터 ──────────────────────────────────────────────────────
#
#   project: sy-0602-grok  workspace: oksusu
#
#   [필수 4개 태그로 정확히 찾기]
#   model:grok/grok-4.3
#   dataset_index:S2_FIN_NONQUANT_MULTI/4f87bbef
#   prompt_hash:02d9caddc6                         ← QA system prompt
#   ontology_hash:799a63e842                       ← medium arm
#
#   [포맷 실험 필터]
#   experiment:serialization                       ← 이 실험 전체
#   format:table                                   ← table 포맷만
#   format:nl                                      ← natural language만
#   slice:S2_FIN_NONQUANT_MULTI                    ← 슬라이스 필터
#
#   [Feedback scores 컬럼 (정렬 가능)]
#   number_overlap  contains_match  judge_score (backfill 후)
#
#   [trace name 패턴]
#   {format}/{case_id}/graph    예: table/4f87bbef/graph
#
# ── Neo4j Browser / Cypher ───────────────────────────────────────────
#
#   DB: finderquick (bolt://34.226.142.183:7687)
#
#   -- 케이스별 도메인 노드 보기 --
#   MATCH (n {_workspace_id: "quick-v1-medium-4f87bbef"})
#   WHERE NOT n:Document AND NOT n:DocumentVersion
#     AND NOT n:Chunk AND NOT n:Section
#   RETURN labels(n) AS type, n.name AS name, n.value AS value,
#          n.period AS period, n.basis AS basis
#   ORDER BY type, name
#
#   -- 관계 보기 --
#   MATCH (a {_workspace_id: "quick-v1-medium-4f87bbef"})-[r]->(b)
#   WHERE NOT a:Document AND NOT a:Chunk
#   RETURN a.name AS from, type(r) AS rel, b.name AS to
#   LIMIT 30
#
#   -- 전체 12 케이스 도메인 노드 수 비교 --
#   MATCH (n)
#   WHERE n._workspace_id STARTS WITH "quick-v1-medium"
#     AND NOT n:Document AND NOT n:DocumentVersion
#     AND NOT n:Chunk AND NOT n:Section
#   WITH n._workspace_id AS ws, count(n) AS domain_nodes
#   RETURN ws, domain_nodes ORDER BY ws
#
#   -- 슬라이스별 엔티티 타입 분포 (S2 케이스) --
#   MATCH (n)
#   WHERE n._workspace_id IN [
#     "quick-v1-medium-4f87bbef", "quick-v1-medium-347bdeaa",
#     "quick-v1-medium-92a58726", "quick-v1-medium-dc4dc72e"
#   ] AND NOT n:Document AND NOT n:DocumentVersion
#     AND NOT n:Chunk AND NOT n:Section
#   RETURN labels(n)[0] AS entity_type, count(n) AS cnt
#   ORDER BY cnt DESC
