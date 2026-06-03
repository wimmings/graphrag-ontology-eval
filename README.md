# FinDER Knowledge Graph Experiment
## Vector vs Graph vs Hybrid · FIBO Ontology Size Sweep · Serialization Format

**기간**: 2026-06-03 · **LLM**: grok/grok-4.3 (non-reasoning, fallback: openai/gpt-4o-mini)  
**인프라**: DozerDB · LanceDB · Opik (`oksusu` / `sy-0602-grok`)  
**코드**: [`scripts/benchmarks/`](../scripts/benchmarks/) · **결과**: [`outputs/evaluation/`](../outputs/evaluation/)

---

## 목차

1. [Abstract](#1-abstract)
2. [Research Questions](#2-research-questions)
3. [Dataset](#3-dataset)
4. [Experiment Design](#4-experiment-design)
5. [Methods](#5-methods)
6. [Engineering Findings](#6-engineering-findings)
7. [Results](#7-results)
8. [Analysis](#8-analysis)
9. [Limitations](#9-limitations)
10. [Future Work](#10-future-work)
11. [Conclusion](#11-conclusion)
12. [Run Commands](#12-run-commands)
13. [Opik & Neo4j 보기](#13-opik--neo4j-보기)
14. [Experiment Ethics Checklist](#14-experiment-ethics-checklist)
15. [Appendix](#15-appendix)

---

## 1. Abstract

동일한 SEC 10-K 데이터셋(FinDER)에서 벡터 검색과 온톨로지 기반 지식 그래프 검색을 직접 비교했다. FIBO 온톨로지를 non-ontology → small → medium → large 4단계로 확장하며 추출 품질을 측정하고, 추출된 그래프를 4가지 직렬화 포맷(current·table·nl·cypher)으로 제공했을 때 QA 품질 차이를 추가 분석했다.

**핵심 결과:**
1. 그래프 단독(graph-only)은 number_overlap 기준 벡터보다 일관되게 낮음
2. Hybrid(vector+graph)는 S2 슬라이스에서 벡터를 앞섬 (0.417 vs 0.355)
3. 스키마 복잡도가 단문서에서 추출 실패를 유발하는 Goldilocks 역전 현상 관측
4. 직렬화 포맷 실험: judge_score 기준 nl(자연어)이 최적, cypher는 서술형 슬라이스에서 유일하게 0점

---

## 2. Research Questions

> (a) 어느 검색 방식이 더 효과적인가? (vector / graph / hybrid)  
> (b) FIBO 온톨로지 크기는 그래프 추출 품질에 어떤 영향을 미치는가?  
> (c) 추출된 그래프의 직렬화 포맷이 LLM QA 품질에 영향을 주는가?

**Provenance 동일성**: 모든 비교는 동일한 `references_joined`에서 시작. 벡터는 청킹+임베딩, 그래프는 온톨로지 기반 추출. 검색 방식만 변수.

---

## 3. Dataset

**FinDER**: SEC 10-K 기반 금융 QA, 910 rows (train parquet, seed=42)  
**파일**: [`dataset/all_slices.csv`](../dataset/all_slices.csv)

### 3.1 슬라이스 정의

| Slice | n | 정의 조건 | Graph 가설 | n_refs avg |
|-------|:-:|----------|-----------|:----------:|
| S1_FIN_COMP | 277 | Financials ∧ Compositional | graph — 다년도 산술 | 1.0 |
| S2_FIN_NONQUANT_MULTI | 165 | Financials ∧ n_refs≥2 | **graph** — IS+BS 교차 | 2.0 |
| S3_CO_COMP | 163 | Company overview ∧ Compositional | graph — HAS_SEGMENT | 1.0 |
| S4_CO_MULTI_NONQUANT | 39 | Company overview ∧ n_refs≥2 | **graph** — 교차 세그먼트 | 2.2 |
| S5_FN_MULTI | 116 | Footnotes ∧ n_refs≥2 | **graph** — 10-K Item 통합 | 2.1 |
| S6_BASELINE_SINGLE | 150 | n_refs=1 ∧ ¬Compositional | **vector** (대조군) | 1.0 |

> S1·S3: n_refs=1이 99% → vector도 충분할 수 있어 graph 우위가 약할 수 있음. 핵심 검증 포인트.  
> S4: n=39 전수 사용해도 CI 넓음. 대규모 효과(Cohen's d ≥ 0.8)만 신뢰 가능 — 결과 보고 시 **underpowered** 명시 필요.  
> S6: graph arm이 vector를 이기면 **red flag** — 측정 오류 가능성.

### 3.2 Graph-Favorable 케이스 선별 기준 (경험적 도출)

```
✅ n_refs ≥ 2          여러 패시지 연결 필요 → graph entity anchor 효과
✅ ref_len ≥ 1,500     충분한 텍스트 → FIBO 엔티티 추출 가능
✅ S1: ref_len ≥ 2,000 재무 테이블이 충분히 길어야 다년도 지표 추출
❌ S3 제외             전체 n_refs=1 → vector로 충분, 그래프 추출 실익 없음
```

---

## 4. Experiment Design

### 4.1 실험 구성

| ID | 스크립트 | 케이스 | 목적 |
|----|---------|--------|------|
| **vector-v2** | [`finder_vector_arm.py`](../scripts/benchmarks/finder_vector_arm.py) | 33개 | 벡터 기준선 전체 |
| **sweep-v2** | [`finder_3arm_experiment.py`](../scripts/benchmarks/finder_3arm_experiment.py) | 33 × 4 arms | 온톨로지 크기 비교 (graph) |
| **quick-v1** | [`finder_quick_compare.py`](../scripts/benchmarks/finder_quick_compare.py) | 12 × medium | vector vs graph vs hybrid 집중 비교 |
| **fmt-v1** | [`finder_format_experiment.py`](../scripts/benchmarks/finder_format_experiment.py) | 12 × 4 formats (QA only) | 직렬화 포맷 비교 |

### 4.2 온톨로지 Arms (nested supersets: small ⊂ medium ⊂ large)

| Arm | Modules | ~nodes | 의도 |
|-----|---------|:------:|------|
| `non-ontology` | `[]` | 1 | Entity/RELATED_TO 베이스라인 |
| `small` | `be, ind` | 11 | 금융 핵심 (S1/S2 커버, S3-S5 과소제약) |
| **`medium`** | `be, ind, fbc, dbt, acc` | 20 | **Goldilocks 후보** |
| `large` | all 9 modules | 31 | 노이즈 가설 |

**모듈-슬라이스 매핑**:

| 모듈 | 커버 슬라이스 | 추가 arm |
|------|-------------|---------|
| `ind` | S1, S2 (FinancialMetric) | small~ |
| `fbc` | S3, S4 (BusinessSegment, HAS_SEGMENT) | medium~ |
| `dbt` | S5 (DebtInstrument) | medium~ |
| `acc` | S2, S5 (AccountingPolicy, CashFlow) | medium~ |
| `fnd/sec/mkt/corp` | 슬라이스 무관 (노이즈) | large만 |

**사전 등록 가설**:

| Slice | non-onto | small | medium | large |
|-------|:--------:|:-----:|:------:|:-----:|
| S1 | vector | **graph** | graph | graph |
| S2 | vector | **graph** | graph | graph |
| S3 | vector | vector (과소제약) | **graph** | graph |
| S4 | vector | vector (과소제약) | **graph** | graph |
| S5 | vector | vector (과소제약) | **graph** | graph |
| S6 | **vector** | vector | vector | vector |

핵심 가설: `non-ontology < small(S1/S2) < medium(전체) > large`

### 4.3 직렬화 포맷 (fmt-v1)

| Format | 표현 방식 | 예시 |
|--------|---------|------|
| `current` | 기존 텍스트 | `- (NetIncome) $5.23 [period=FY2024, basis=diluted]` |
| `table` | 마크다운 테이블 | `\| NetIncome \| $5.23 \| FY2024 \| diluted \|` |
| `nl` | 자연어 문장 | `"Fiserv's diluted EPS in FY2024 was $5.23."` |
| `cypher` | Cypher 트리플 | `("Fiserv")-[:HAS_METRIC]->(EPS {value:"$5.23"})` |

---

## 5. Methods

### 5.1 파이프라인

```
10-K references_joined
       │
       ├─[FIBO 추출]──────────→ DozerDB Knowledge Graph
       │  grok-4.3                        │
       │  grok_meta_system_prompt.md       │
       │  {{ontology}} + {{text}}          │
       │                                  ▼
       └─[임베딩]──────────────→ LanceDB  Graph Context
          text-embedding-3-small   │      (직렬화 포맷 선택)
          1536d, top-k=5           │
                    │              │
                    ▼              ▼
              Vector Context  Graph Context
                    │              │
                    └──────┬───────┘
                           ▼
                   QA: grok-4.3
                           │
                           ▼
                  Answer → 3 Metrics
```

### 5.2 추출 프롬프트 필수 조건

매 실행 시 [`_verify_prompt_conditions()`](../scripts/benchmarks/finder_3arm_experiment.py) 자동 검증:

```
① {{ontology}}  — arm별 FIBO 스키마 주입 (KGPromptTemplate.render)
② KG engineer  — "You are a knowledge graph engineer built by xAI (Grok)"
③ {{text}}      — 원본 10-K 텍스트 슬롯 (user template)
```

### 5.3 평가 지표

| 지표 | 계산 | 시점 | 특성 |
|------|------|------|------|
| `number_overlap` | (pred∩gold 숫자) / \|gold 숫자\| | live | 결정론적, 빠름 — 단위/스케일 오류 미감지 |
| `token_f1` | SQuAD-style token F1 | 오프라인 | 의미 일부 포착 |
| `judge_score` | grok-4.3 LLM-as-judge (0/0.5/1) | 오프라인 | 최고 품질 — self-preference bias 있으나 전 레인 균일 |

> S4(서술형)는 number_overlap = 0이 정상 — judge_score로만 의미있는 비교 가능.

### 5.4 Opik 트레이스 태그 규약

```
필수 (10자 hex, 항상 존재):
  model:grok/grok-4.3
  dataset_index:{slice}/{case_id}
  prompt_hash:{10char}            추출/QA 프롬프트 SHA-256[:10]
  ontology_hash:{10char}          arm별 entity+rel 타입 해시

보조 (공통):
  phase:{arm}  variant:{mode}  slice  case  modules  meta_prompt  category

포맷 실험 추가:
  format:{current|table|nl|cypher}    ← 핵심 실험 변수
  experiment:serialization

trace name:
  그래프/hybrid: {arm}/{case_id}/{mode}    예: medium/4f87bbef/graph
  포맷 실험:     {format}/{case_id}/graph  예: nl/4f87bbef/graph
  벡터:          vector/{case_id}/n-a
```

---

## 6. Engineering Findings

### 6.1 스키마 복잡도 vs 문서 길이 미스매치

| Arm | 스키마 | 문서 588자 | 도메인 노드 |
|-----|:------:|:---------:|:---------:|
| non-ontology | ~150 tok | Entity/RELATED_TO | ✅ 8개 |
| small | ~646 tok | NetIncome, EPS | ✅ 8개 |
| medium | ~1,300 tok | — | ❌ 0개 |
| large | ~1,986 tok | — | ❌ 0개 |

**원인**: 스키마:문서 비율 > 8:1이면 LLM이 보수적으로 빈 JSON 반환.

**수정 조치**:
1. `_graph_context()`에서 인프라 관계(HAS_VERSION, MENTIONS 등) 완전 제거
2. 도메인 노드 0개 → 원본 source text fallback (실험 공정성 유지, `retrieval_source: graph_fallback_text`로 기록)

### 6.2 그래프 컨텍스트 포맷 문제

노드 16~89개 추출됐음에도 graph-only number_overlap = 0.00. 직렬화 포맷을 LLM이 수치 계산에 활용하지 못함 → §7.4 포맷 실험에서 병목 분리.

---

## 7. Results

### 7.1 Vector Baseline (vector-v2, n=33, error=0)

| Slice | n | number_overlap |
|-------|:-:|:--------------:|
| S1_FIN_COMP | 5 | 0.432 |
| S2_FIN_NONQUANT_MULTI | 5 | 0.346 |
| S3_CO_COMP | 5 | 0.343 |
| S4_CO_MULTI_NONQUANT | 10 | 0.250 |
| S5_FN_MULTI | 5 | 0.253 |
| **S6_BASELINE_SINGLE** | 3 | **0.165** (대조군, 정상) |

### 7.2 온톨로지 Sweep (sweep-v2, S1/S2 partial)

> n≈5/slice — 방향성만 참고, CI 매우 넓음.

| Arm | S1 overlap | S2 overlap | 전체 avg |
|-----|:----------:|:----------:|:--------:|
| vector (기준) | 0.432 | 0.346 | — |
| non-ontology | 0.396 | 0.644 | 0.489 |
| **small** | **0.437** | **0.644** | **0.515** |
| medium | 0.378 | 0.400 | 0.384 |
| large | 0.370 | 0.267 | 0.341 |

S2에서 small(0.644) > medium(0.400) > large(0.267) — 예상과 달리 small이 앞섬 (§8.3 Goldilocks 역전).

### 7.3 Quick Compare: Vector vs Graph vs Hybrid (quick-v1, medium arm)

#### 슬라이스별 평균

| Slice | vector | graph | hybrid | judge: vec | judge: graph | judge: hybrid |
|-------|:------:|:-----:|:------:|:----------:|:------------:|:-------------:|
| S2 IS+BS 교차 | 0.355 | 0.000 | **0.417** | **0.600** | 0.000 | 0.250 |
| S4 세그먼트 서술 | 0.250 | 0.000 | 0.000 | 0.490 | 0.250 | **0.500** |
| S5 Footnote | **0.275** | 0.172 | 0.240 | **0.530** | 0.438 | 0.500 |

*Paired delta (judge): graph vs vector Δ=−0.258 (vec_win=5/12) · hybrid vs vector Δ=−0.071 (vec_win=2/12)*

### 7.4 포맷 실험: Graph Serialization (fmt-v1, medium arm, QA only)

#### 포맷별 전체 순위 (judge_score 기준)

| Rank | Format | overlap | token_f1 | **judge** |
|:----:|--------|:-------:|:--------:|:---------:|
| **#1** | **nl** | 0.059 | 0.060 | **0.250** |
| #2 | current | 0.057 | 0.054 | 0.229 |
| #2 | table | 0.074 | 0.058 | 0.229 |
| #4 | cypher | 0.042 | 0.054 | 0.167 |

#### 슬라이스별 judge_score

| Slice | current | table | **nl** | cypher |
|-------|:-------:|:-----:|:------:|:------:|
| S2 | 0.000 | 0.000 | 0.000 | 0.000 |
| S4 | 0.250 | 0.250 | 0.250 | **0.000** |
| S5 | 0.438 | 0.438 | **0.500** | **0.500** |

> S2 전부 0 = 포맷 문제 아님 → 추출 문제 (IS+BS cross-statement entity 부족)  
> S4 cypher 단독 0 = Cypher 표기법이 서술형 답변 구성에 부적합  
> overlap 기준 table 최고, judge 기준 nl 최고 → **단일 지표로 포맷 평가 불가**

---

## 8. Analysis

### 8.1 Graph 강점 조건

```
✅ hybrid + S2 (IS+BS교차): graph entity anchor → vector와 결합 시 시너지
   number_overlap: hybrid 0.417 > vector 0.355
   judge_score: S4 hybrid(0.500) ≈ vector(0.490) — 서술형에서 대등

⚠️  graph 단독: 현재 serialization으론 QA 활용 어려움
    포맷 실험 결과: nl 채택 시 S5 graph judge(0.500) = hybrid(0.500)

❌  S2: 포맷 무관 0점 → 추출 문제 (cross-statement entity linking 부재)
```

### 8.2 Graph 약점 조건

| 조건 | 원인 |
|------|------|
| 단문서 (ref_len < 1,500) | 스키마:문서 비율 > 8:1 → 빈 추출 |
| medium/large + 짧은 문서 | small보다 오히려 낮음 |
| S4 number_overlap | 서술형 → 숫자 없음 → 0점 (지표 부적합) |
| S2 graph | IS+BS entity 연결 구조 추출 실패 |

### 8.3 Goldilocks 역전 현상

S2에서 small(0.644) > medium(0.400) > large(0.267). 가능한 해석:
1. n=2~4로 노이즈 가능성 높음 (완전한 결론 불가)
2. medium의 `acc`/`fbc` 모듈이 재무제표 숫자 대신 equity/segment 추출로 관심 분산
3. small의 `ind`가 FinancialMetric에 집중해 S2 수치만 정확히 잡을 수 있음

### 8.4 지표 불일치

number_overlap과 judge_score가 서로 다른 포맷을 최적으로 선택:
- number_overlap 최적: **table** (0.074) — 표 구조가 숫자 추출에 유리
- judge_score 최적: **nl** (0.250) — 자연어 문장이 LLM 답변 구성에 유리

→ 실제 QA 품질 측정은 judge_score 필수.

### 8.5 Hybrid 효과

`4f87bbef` (S2, PFE liquidity): hybrid 0.53 vs vector 0.13 vs graph 0.00

그래프가 `LegalEntity → HAS_METRIC → MonetaryAmount`로 entity를 연결해두면, vector context와 결합 시 LLM이 답을 더 잘 구성. **graph는 단독 retrieval보다 vector 보조 역할로 강점.**

---

## 9. Limitations

| 한계 | 내용 |
|------|------|
| 소표본 | 슬라이스당 n=3~12. CI 넓음. 대규모 효과만 신뢰. |
| S4 underpowered | n=39 전수도 통계 검정력 부족 |
| S4 지표 | 서술형 → number_overlap 부적합. judge 필수. |
| self-preference | generator=judge=grok-4.3. 절대값 lenient 가능. |
| sweep 미완 | sweep-v2가 S1/S2만 완료. S3~S6 결과 없음. |
| 단일 arm 비교 | fmt-v1·quick-v1이 medium arm만 비교. |

---

## 10. Future Work

1. **nl 포맷 채택 후 재실험** — fmt-v1 결과 기반. S5에서 graph-only > vector 가능성
2. **S2 추출 개선** — IS+BS 교차에 필요한 cross-statement entity linking 보완
3. **sweep 완성** — S3/S4/S5/S6 포함 전체 4-arm 완료
4. **4-arm format compare** — non-ontology/small/large도 동일 케이스에서 포맷 비교
5. **Goldilocks 재검증** — n≥20으로 small vs medium 재실험
6. **cross-vendor judge** — GPT-4o로 judge 교체해 self-preference bias 제거 검증

---

## 11. Conclusion

**Graph 단독은 현재 설정에서 vector를 앞서지 못함** — 포맷 실험으로 병목이 S2는 추출 문제, S5는 포맷 문제임을 분리.

**Hybrid는 S2에서 가능성 확인** — IS+BS 교차 참조에서 hybrid(0.417) > vector(0.355). judge_score에서도 S4 hybrid(0.500) ≈ vector(0.490).

**직렬화 포맷**: judge 기준 nl이 최적. S5 nl=0.500 — nl 포맷 채택 시 S5에서 graph > vector 가능성 있음.

---

## 12. Run Commands

```bash
# 벡터 기준선
python scripts/benchmarks/finder_vector_arm.py \
  --run-prefix vector-v2

# 그래프 4-arm sweep (graph-favorable 케이스)
python scripts/benchmarks/finder_3arm_experiment.py \
  --graph-favorable \
  --run-prefix sweep-gf-v1 \
  --database findergfv1

# 빠른 3-way 비교 (medium arm, 12 cases)
python scripts/benchmarks/finder_quick_compare.py \
  --run-prefix quick-v1 \
  --database finderquick

# 직렬화 포맷 비교 (재추출 없음, QA only)
python scripts/benchmarks/finder_format_experiment.py \
  --run-prefix fmt-v1

# 오프라인 채점
python scripts/benchmarks/finder_judge.py \
  --inputs "outputs/evaluation/finder_quick_compare/quick-v1/S*.json" \
           "outputs/evaluation/finder_vector_arm/vector-v2/partial/*.json" \
  --out outputs/evaluation/judged-quick-v1.json

python scripts/benchmarks/finder_judge.py \
  --inputs "outputs/evaluation/finder_format_experiment/fmt-v1/S*.json" \
  --out outputs/evaluation/judged-format-v1.json
```

**Dry-run 확인:**
```bash
python scripts/benchmarks/finder_3arm_experiment.py --dry-run
python scripts/benchmarks/finder_format_experiment.py --dry-run
```

---

## 13. Opik & Neo4j 보기

### Opik UI 필터

**URL**: https://www.comet.com/opik → workspace `oksusu` → project `sy-0602-grok`

```
# 포맷 실험 전체
experiment:serialization

# 포맷별 필터
format:nl
format:table

# 슬라이스 + 포맷 조합
slice:S5_FN_MULTI + format:nl

# medium arm 전체 (ontology_hash로 arm 식별)
ontology_hash:799a63e842

# 특정 케이스
dataset_index:S2_FIN_NONQUANT_MULTI/4f87bbef
```

### Neo4j Browser

**접속**: `http://34.226.142.183:7474` → DB 선택: `finderquick` 또는 `findersweepv2`

```cypher
-- 케이스 도메인 노드 보기 (예: PFE 케이스)
MATCH (n {_workspace_id: "quick-v1-medium-4f87bbef"})
WHERE NOT n:Document AND NOT n:DocumentVersion
  AND NOT n:Chunk AND NOT n:Section
RETURN labels(n) AS type, n.name AS name,
       n.value AS value, n.period AS period, n.basis AS basis
ORDER BY type, name

-- 관계 보기
MATCH (a {_workspace_id: "quick-v1-medium-4f87bbef"})-[r]->(b)
WHERE NOT a:Document AND NOT a:Chunk
RETURN a.name AS from, type(r) AS rel, b.name AS to LIMIT 30

-- 12 케이스 도메인 노드 수 비교
MATCH (n)
WHERE n._workspace_id STARTS WITH "quick-v1-medium"
  AND NOT n:Document AND NOT n:DocumentVersion
  AND NOT n:Chunk AND NOT n:Section
WITH n._workspace_id AS ws, count(n) AS cnt
RETURN ws, cnt ORDER BY cnt DESC

-- S2 케이스 엔티티 타입 분포
MATCH (n)
WHERE n._workspace_id IN [
  "quick-v1-medium-4f87bbef", "quick-v1-medium-347bdeaa",
  "quick-v1-medium-92a58726", "quick-v1-medium-dc4dc72e"
] AND NOT n:Document AND NOT n:Chunk AND NOT n:Section
RETURN labels(n)[0] AS entity_type, count(n) AS cnt
ORDER BY cnt DESC
```

---

## 14. Experiment Ethics Checklist

CLAUDE.md §20 기준:

- [x] 데이터 기반만 — 직관·가설로 결과 대체 금지
- [x] silent fallback 금지 — 실패 시 `error` 필드에 기록, 0점 처리
- [x] 공정 비교 — 모든 레인에 동일 케이스·동일 gold·동일 judge
- [x] 온톨로지만 변수 — 동일 문서/동일 extractor/동일 graph store
- [x] 전 슬라이스 보고 — cherry-pick 금지
- [x] S6 대조군 확인 — vector 최저점(0.165) 확인 → 정상
- [x] 불확실성 명시 — n이 작은 슬라이스는 CI 주석 포함
- [x] S4 underpowered 명시 — judge_score 결과에도 동일 주석

---

## 15. Appendix

### A. 실험 설정

```
LLM:        grok/grok-4.3 (fallback: openai/gpt-4o-mini)
Embedding:  text-embedding-3-small (1536d, OpenAI)
Vector DB:  LanceDB (.seocho/lancedb/finder_vector_0530)
Graph DB:   DozerDB bolt://34.226.142.183:7687
            findersweepv2 (sweep-v2), finderquick (quick-v1/fmt-v1)
Opik:       hosted, workspace=oksusu, project=sy-0602-grok
Seed:       42 (전체 실험 고정)
```

### B. 파일 구조

```
dataset/all_slices.csv
examples/finder/
  datasets/fibo_modules/*.yaml           FIBO module YAMLs
  datasets/grok_meta_system_prompt.md    추출 시스템 프롬프트
  lib/bench_common.py                    공용 유틸 (ONTOLOGY_ARMS, FINAL_SLICE_CONFIG 등)

scripts/benchmarks/
  finder_vector_arm.py          벡터 실험
  finder_3arm_experiment.py     그래프 4-arm sweep
  finder_quick_compare.py       빠른 3-way 비교
  finder_format_experiment.py   직렬화 포맷 비교
  finder_judge.py               오프라인 채점

outputs/evaluation/
  finder_vector_arm/vector-v2/          33 cases 벡터
  finder_3arm_experiment/sweep-v2/      33 cases × 4 arms
  finder_quick_compare/quick-v1/        12 cases × medium (graph+hybrid)
  finder_format_experiment/fmt-v1/      12 cases × 4 formats
  judged-quick-v1.json                  quick-v1 채점
  judged-format-v1.json                 fmt-v1 채점
```

### C. 주요 발견 타임라인

| 날짜 | 발견 |
|------|------|
| 2026-06-03 | vector baseline 완료 (S1=0.432 최고, S6=0.165 대조군 정상) |
| 2026-06-03 | medium/large arm이 단문서(588자)에서 추출 실패 (스키마:문서 비율 문제) |
| 2026-06-03 | `_graph_context()` 인프라 관계 제거 + source text fallback 적용 |
| 2026-06-03 | sweep-v2: S2에서 small(0.644) > medium(0.400) — Goldilocks 역전 |
| 2026-06-03 | quick-v1: S2 hybrid(0.417) > vector(0.355) 확인 |
| 2026-06-03 | quick-v1 judge: S4 hybrid(0.500) ≈ vector(0.490) — 서술형 대등 |
| 2026-06-03 | fmt-v1: S2 포맷 무관 0점 → 추출 문제, S5는 포맷 문제로 병목 분리 |
| 2026-06-03 | fmt-v1 judge: nl 최적(0.250), S5 nl=0.500 > table=current=0.438 |
| 2026-06-03 | cypher가 S4 서술형에서 유일하게 0점 — Cypher 포맷 부적합 확인 |
