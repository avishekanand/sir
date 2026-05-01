# RAGtune SVG Figures — Design Spec
**Date:** 2026-05-01  
**Status:** Approved  
**Context:** Create hand-crafted SVG figures for (A) the RAGtune standalone presentation and (B) a 2-slide "we're cooking" section at the end of the Booking.com 2026 talk.

---

## Background

The Booking 2026 talk ("From Pipelines to Self-Improving Retrieval") covers five research systems — QUAM, SUNAR, ORE, ReformIR, CASE — each with its own SVG figure set. At the end of the talk, 2 slides present RAGtune as the concrete software implementation of all five systems.

Separately, the RAGtune standalone marp deck (`docs/presentations/ragtune-feedback-driven-retrieval-group-talk.md`) needs its own figure set so it can be presented independently with the same visual language.

---

## Visual System

All figures must conform exactly to the shared visual system. No deviations.

**Canvas:** `viewBox="0 0 1100 720"`, `background #faf7f2`, `xmlns="http://www.w3.org/2000/svg"`

**Palette:**

| Token | Hex | Usage |
|-------|-----|-------|
| paper | `#faf7f2` | Background |
| ink | `#1a1a1a` | Primary text, boxes, axes |
| accent | `#c8553d` | Feedback arrows, protagonist signal, footer punch line |
| cool | `#2a4747` | Reference systems, teacher signals, teal boxes |
| muted | `#6b665e` | Captions, secondary labels |
| gray-light | `#b8b3a8` | Baseline bars, de-emphasised elements |
| gray-dark | `#8a857a` | Tick marks, secondary borders |
| grid | `#e8e2d6` | Chart grid lines, dividers |

**Typography:** `Georgia, 'Times New Roman', serif` — in both SVG and marp CSS.

- Title: 32px, weight 400, letter-spacing 0.2, fill ink
- Subtitle: 16px italic, fill muted, letter-spacing 0.3
- Body callout: 13–15px
- Small-caps category labels: 11–13px bold, letter-spacing 0.5–2.0

**Standard header block (copy into every figure):**
```svg
<text x="80" y="68" font-size="32" font-weight="400" fill="#1a1a1a" letter-spacing="0.2">Figure Title</text>
<text x="80" y="94" font-size="16" font-style="italic" fill="#6b665e" letter-spacing="0.3">Italic subtitle</text>
<line x1="80" y1="112" x2="200" y2="112" stroke="#1a1a1a" stroke-width="1.2"/>
```

**Standard footer block (when a punch line is needed):**
```svg
<line x1="80" y1="630" x2="200" y2="630" stroke="#c8553d" stroke-width="1.5"/>
<text x="80" y="660" font-size="16" fill="#1a1a1a">Supporting sentence.</text>
<text x="80" y="685" font-size="14" fill="#c8553d" font-weight="700">The one thing to remember.</text>
```

**Arrowhead defs (include in every figure):**
```svg
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
          markerWidth="9" markerHeight="9" orient="auto">
    <path d="M0,0 L10,5 L0,10 z" fill="#1a1a1a"/>
  </marker>
  <marker id="arrowAccent" viewBox="0 0 10 10" refX="9" refY="5"
          markerWidth="9" markerHeight="9" orient="auto">
    <path d="M0,0 L10,5 L0,10 z" fill="#c8553d"/>
  </marker>
</defs>
```

**Arrow semantics:**
- Solid black, stroke-width 2.0–2.4 = forward data flow
- Dashed accent (`stroke-dasharray="8,5"`), stroke-width 2.4–3.0 = feedback / return path
- Faint gray dashed = absent / broken path (used in pipeline failure figure)

**Other rules:**
- Box corner radius: 4px (`rx="4"`)
- No gradients, no shadows, no CSS animations (static step files only)
- Dimmed past elements: `opacity="0.35"` minimum — never lower
- Highlight region: `fill="#2a4747" fill-opacity="0.05" rx="2"`

---

## File Organisation

### Part A — RAGtune standalone

**Location:** `ragtune/docs/presentations/ragtune-diagrams/`

**Marp deck:** `ragtune/docs/presentations/ragtune-feedback-driven-retrieval-group-talk.md`  
Update figure slides to use full-bleed pattern:
```markdown
---
<!-- _header: '' -->
<!-- _footer: '' -->

![bg fit](ragtune-diagrams/slide_NAME_s1.svg)
```

### Part B — Booking talk

**Location:** `talks/booking-2026/booking-diagrams/`

**Marp deck:** `talks/booking-2026/booking-2026.md` (to be created)  
Same full-bleed pattern.

---

## Figure Index

### Part A — RAGtune standalone (4 figures, 12 SVGs)

#### A1 — `slide_rag_pipeline`
**Title:** "Standard RAG is a one-way street"  
**Type:** architecture

| Step | Shows |
|------|-------|
| s1 | Three boxes: **Retriever** (cool) · **Reranker** (ink) · **LLM** (ink). No arrows. |
| s2 | Solid black forward arrows. Label: "N = 1000 candidates" above Retriever, "top-K" above Reranker. Gray candidate-pool dot cluster between Retriever and Reranker. |
| s3 | Broken return path: faint gray dashed arrow from LLM back toward Retriever with ✗. Footer: *"No feedback path. Every query treated identically."* |

---

#### A2 — `slide_ragtune_loop`
**Title:** "RAGtune: The Loop"  
**Type:** architecture

| Step | Shows |
|------|-------|
| s1 | Four boxes in clockwise cycle: **CandidatePool** (cool, top-left) · **Estimator** (ink, top-right) · **Scheduler** (ink, bottom-right) · **Reranker** (ink, bottom-left). No arrows. Subtitle: "Budget-aware iterative reranking". |
| s2 | Solid black arrows clockwise: Pool→Estimator→Scheduler→Reranker→Pool. Arrow labels: "priorities" · "batch" · "scores" · "update". |
| s3 | Dashed accent arrow from Reranker back to Estimator, labelled "signal". Small **CostTracker** progress bar top-right. Footer: *"Same budget. Scheduling driven by feedback, not retrieval rank."* |

---

#### A3 — `slide_estimator_matrix`
**Title:** "Estimator: converting feedback into priorities"  
**Type:** concept

| Step | Shows |
|------|-------|
| s1 | 2×2 axis grid. X: *feedback richness* (none → rich). Y: *signal type* (structural → semantic). Grid lines only. Subtitle: "Which estimator for which situation?" |
| s2 | 4 labeled points: **Baseline** (bottom-left, gray) · **Utility** (mid-left, muted) · **Similarity** (top-mid, ink) · **ReformIR** (top-right, cool). Short descriptor per point. |
| s3 | Diagonal teal band bottom-left → top-right: "sweet spot — use as much feedback as you have". Accent circle around ReformIR. Footer: *"ReformIR: after 3 scored docs, source weights update every iteration."* |

---

#### A4 — `slide_results_pareto`
**Title:** "Results: the Pareto picture"  
**Type:** result chart

| Step | Shows |
|------|-------|
| s1 | Axes only. X: *avg rerank docs (budget)*. Y: *avg NDCG@5*. Grid lines (#e8e2d6). Subtitle: "3 datasets · MonoT5 reranker". |
| s2 | Four gray bars: **BM25** (0 docs, 0.656) · **tight/5** (0.720) · **medium/15** (0.749) · **loose/30** (0.748). Labels above. Medium ≈ loose height is deliberate — "more docs ≠ better". |
| s3 | Accent bar: **convergence feedback / 10 docs** (0.760) — taller than loose/30. Delta annotation in accent: "+4% NDCG, ⅓ the budget". Footer: *"The ceiling isn't the budget — it's how you use it."* |

*Bar heights: `y = base - scale * value` where base = 580, scale = 600. Values: BM25=0.656, tight=0.720, medium=0.749, loose=0.748, feedback=0.760. Swap in real paper values by editing `value` and recomputing y.*

---

### Part B — Booking talk "we're cooking" (2 figures, 6 SVGs)

#### B1 — `slide_ragtune_mapping`
**Title:** "RAGtune: from science to software"  
**Type:** architecture

| Step | Shows |
|------|-------|
| s1 | Top row: 5 cool/teal boxes — **QUAM** · **SUNAR** · **ORE** · **ReformIR** · **CASE**. Subtitle: "Five algorithms. One loop." |
| s2 | Below: RAGtune controller loop with 4 labeled component slots — **Estimator** · **Scheduler** · **Reranker** · **Pool**. No connections yet. |
| s3 | Dashed accent arrows: QUAM→Estimator, ORE→Scheduler, ReformIR→Estimator, CASE→Estimator. SUNAR gets a single accent arrow pointing to the entire RAGtune loop group (not a specific slot) — it implements a complete retrieve-rerank-learn cycle, so it maps to the loop as a whole. Footer: *"RAGtune is the software that makes the science runnable."* |

---

#### B2 — `slide_ragtune_status`
**Title:** "RAGtune · active development"  
**Type:** callout

| Step | Shows |
|------|-------|
| s1 | Large typographic header: "RAGtune · active development". Subtitle: "github.com/avishekanand/sir". Centered, generous whitespace. |
| s2 | Two columns. **✓ running now** (cool/teal): Controller loop · CandidatePool · Budget enforcement · ReformIR estimator · ActiveLearning scheduler. **⟳ in progress** (muted): QUAM adapter · ORE bandit scheduler · CASE estimator · SUNAR loop. |
| s3 | Footer accent line. Closing line in accent red (14px bold): *"The science is done. The software is being built."* |

---

## Marp Integration

### RAGtune deck changes

Replace these four text slides in the marp deck with step sequences (slide titles as they appear in the `.md`):

| Slide heading to replace | Figure to insert |
|--------------------------|-----------------|
| `# Standard RAG is a one-way street` | `slide_rag_pipeline_s1/s2/s3` |
| `# RAGtune: The Loop` | `slide_ragtune_loop_s1/s2/s3` |
| `# Estimator: converting feedback into priorities` | `slide_estimator_matrix_s1/s2/s3` |
| `# Results: the Pareto picture` | `slide_results_pareto_s1/s2/s3` |

Keep all other text slides unchanged. Each replacement turns one `#` heading slide into three full-bleed SVG slides using this pattern:

```markdown
---
<!-- _header: '' -->
<!-- _footer: '' -->

![bg fit](ragtune-diagrams/slide_rag_pipeline_s1.svg)

---
<!-- _header: '' -->
<!-- _footer: '' -->

![bg fit](ragtune-diagrams/slide_rag_pipeline_s2.svg)

---
<!-- _header: '' -->
<!-- _footer: '' -->

![bg fit](ragtune-diagrams/slide_rag_pipeline_s3.svg)
```

Also update the deck's frontmatter to use the shared visual system CSS (Georgia serif, paper background, CSS tokens matching the SVG palette).

### Booking deck

The 2-slide RAGtune section slots into Act 4 (Vision) of the Booking talk, after the self-play loop slides (around p43). Use the same full-bleed step pattern.

---

## Implementation Notes

- Spawn 3 agents in parallel per batch: each agent owns 1 figure (3 step files)
- Agent prompt must include the full visual system token list from §SVG Agent Prompt Template in `presentation-best-practices.md`
- All numeric values in result charts are placeholders — documented in SVG comments with the formula so they can be swapped for real paper data
- No CSS `@keyframes` — static step files only
- Text must not overflow boxes: use explicit `text-anchor` and pre-wrap at known widths
