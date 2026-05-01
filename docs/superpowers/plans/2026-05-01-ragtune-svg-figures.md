# RAGtune SVG Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create 18 SVG step files (6 figures × 3 steps each) for the RAGtune standalone presentation and the Booking 2026 talk, matching the shared visual system exactly.

**Architecture:** Two independent deliverable sets — Part A (4 figures) goes into `ragtune/docs/presentations/ragtune-diagrams/` for the standalone deck; Part B (2 figures) goes into `talks/booking-2026/booking-diagrams/` for the booking talk. Each figure has 3 complete, self-contained step files (_s1, _s2, _s3). The marp deck is updated at the end to use `![bg fit]` full-bleed step sequences.

**Tech Stack:** Hand-written SVG (no build tools), `xmllint` for XML validation, `rsvg-convert` for optional PNG preview, Marp for the deck.

**Visual system reference:** `docs/superpowers/specs/2026-05-01-ragtune-svg-figures-design.md`  
**Best practices reference:** `/Users/avishekanand/slides/presentation-best-practices.md`  
**Source SVG to adapt (A1, A2):** `/Users/avishekanand/slides/booking-2026/booking-diagrams/figures/slide_p10_cascade_pipeline.svg`

---

## Shared SVG constants (apply to every file)

```
viewBox:    "0 0 1100 720"
background: #faf7f2
font-family: Georgia, 'Times New Roman', serif

Colors:
  ink:        #1a1a1a   (primary text, boxes, axes)
  accent:     #c8553d   (feedback arrows, punch lines)
  cool:       #2a4747   (teal boxes, reference systems)
  muted:      #6b665e   (captions, subtitles)
  gray-light: #b8b3a8   (candidate dots, baseline bars)
  gray-dark:  #8a857a   (tick marks)
  grid:       #e8e2d6   (chart grid lines)
  paper:      #faf7f2   (background)

Standard header (copy into every figure):
  <text x="80" y="68" font-size="32" font-weight="400" fill="#1a1a1a" letter-spacing="0.2">TITLE</text>
  <text x="80" y="94" font-size="16" font-style="italic" fill="#6b665e" letter-spacing="0.3">SUBTITLE</text>
  <line x1="80" y1="112" x2="200" y2="112" stroke="#1a1a1a" stroke-width="1.2"/>

Standard arrowhead defs (copy into every figure):
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

Standard footer (when a punch line is needed):
  <line x1="80" y1="630" x2="200" y2="630" stroke="#c8553d" stroke-width="1.5"/>
  <text x="80" y="658" font-size="16" fill="#1a1a1a">Supporting sentence.</text>
  <text x="80" y="683" font-size="14" fill="#c8553d" font-weight="700">Accent punch line.</text>
```

---

## File Structure

```
ragtune/
  docs/presentations/
    ragtune-diagrams/                      ← CREATE (Part A figures)
      slide_rag_pipeline_s1.svg
      slide_rag_pipeline_s2.svg
      slide_rag_pipeline_s3.svg
      slide_ragtune_loop_s1.svg
      slide_ragtune_loop_s2.svg
      slide_ragtune_loop_s3.svg
      slide_estimator_matrix_s1.svg
      slide_estimator_matrix_s2.svg
      slide_estimator_matrix_s3.svg
      slide_results_pareto_s1.svg
      slide_results_pareto_s2.svg
      slide_results_pareto_s3.svg
    ragtune-feedback-driven-retrieval-group-talk.md   ← MODIFY (4 slides → step sequences)

talks/booking-2026/
  booking-diagrams/                        ← CREATE (Part B figures)
    slide_ragtune_mapping_s1.svg
    slide_ragtune_mapping_s2.svg
    slide_ragtune_mapping_s3.svg
    slide_ragtune_status_s1.svg
    slide_ragtune_status_s2.svg
    slide_ragtune_status_s3.svg
  booking-ragtune.md                       ← CREATE (2-slide booking section)
```

---

## Task 0: Setup

**Files:**
- Create: `ragtune/docs/presentations/ragtune-diagrams/` (directory)
- Create: `talks/booking-2026/booking-diagrams/` (directory)
- Modify: `ragtune/.gitignore`

- [ ] **Step 1: Create output directories**

```bash
mkdir -p /Users/avishekanand/Projects/ragtune/docs/presentations/ragtune-diagrams
mkdir -p /Users/avishekanand/talks/booking-2026/booking-diagrams
```

- [ ] **Step 2: Add .superpowers to .gitignore**

Check `ragtune/.gitignore` — if `.superpowers/` is not present, add it:
```bash
grep -q '.superpowers' /Users/avishekanand/Projects/ragtune/.gitignore || \
  echo '.superpowers/' >> /Users/avishekanand/Projects/ragtune/.gitignore
```

- [ ] **Step 3: Commit setup**

```bash
cd /Users/avishekanand/Projects/ragtune
git add .gitignore
git commit -m "chore: add ragtune-diagrams dir and .superpowers to gitignore"
```

---

## Task 1: A1 — slide_rag_pipeline (Standard RAG pipeline)

**Files:**
- Create: `docs/presentations/ragtune-diagrams/slide_rag_pipeline_s1.svg`
- Create: `docs/presentations/ragtune-diagrams/slide_rag_pipeline_s2.svg`
- Create: `docs/presentations/ragtune-diagrams/slide_rag_pipeline_s3.svg`

**Geometry source:** Adapted from `/Users/avishekanand/slides/booking-2026/booking-diagrams/figures/slide_p10_cascade_pipeline.svg`. Keep all x/y coordinates from p10 verbatim. The only change from p10 is: use `viewBox="0 0 1100 720"` (not `"0 130 1100 480"`), add the standard title header at y=68/94/112, and replace the "THE ASSUMPTION" callout with the footer block.

Pipeline element positions (from p10, y-center = 320):
- Query circle: cx=130, cy=320, r=50
- Retrieve box: x=240, y=280, w=160, h=80, rx=4
- Pool dashed box: x=430, y=225, w=280, h=190, rx=3
- Pool dots: 6 rows × 12 cols, start cx=450 cy=245, spacing=22
- Teal relevant dots inside pool: (494,267) (582,311) (648,333) (538,355), r=4.5
- Pool legend: circle at (438,436) gray-light, circle at (512,436) cool, text at (450,440) and (522,440)
- Rerank box: x=740, y=280, w=140, h=80, rx=4
- Generate box: x=920, y=280, w=130, h=80, rx=4
- Flow arrows: (180→234, y=320), (402→424, y=320), (712→734, y=320), (882→914, y=320)

- [ ] **Step 1: Write slide_rag_pipeline_s1.svg**

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 720"
     font-family="Georgia, 'Times New Roman', serif" width="1100" height="720">
  <!-- slide_rag_pipeline s1: pipeline scaffold with title -->
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

  <rect width="1100" height="720" fill="#faf7f2"/>

  <!-- Header -->
  <text x="80" y="68" font-size="32" font-weight="400" fill="#1a1a1a" letter-spacing="0.2">Standard RAG is a one-way street</text>
  <text x="80" y="94" font-size="16" font-style="italic" fill="#6b665e" letter-spacing="0.3">One pass. No adaptation.</text>
  <line x1="80" y1="112" x2="200" y2="112" stroke="#1a1a1a" stroke-width="1.2"/>

  <!-- Query -->
  <circle cx="130" cy="320" r="50" fill="#faf7f2" stroke="#1a1a1a" stroke-width="2"/>
  <text x="130" y="316" font-size="15" fill="#1a1a1a" text-anchor="middle" font-weight="600">Query</text>
  <text x="130" y="335" font-size="12" fill="#6b665e" text-anchor="middle" font-style="italic">q</text>

  <!-- Retrieve -->
  <rect x="240" y="280" width="160" height="80" fill="#faf7f2" stroke="#1a1a1a" stroke-width="2" rx="4"/>
  <text x="320" y="313" font-size="15" fill="#1a1a1a" text-anchor="middle" font-weight="600">Retrieve</text>
  <text x="320" y="335" font-size="12" fill="#6b665e" text-anchor="middle" font-style="italic">embeddings</text>

  <!-- top-1000 pool -->
  <rect x="430" y="225" width="280" height="190" fill="none" stroke="#6b665e"
        stroke-width="1" stroke-dasharray="4,4" rx="3"/>
  <text x="570" y="218" font-size="13" fill="#6b665e" text-anchor="middle"
        font-weight="600" letter-spacing="0.3">top-1000 candidates</text>

  <!-- candidate dots: 6 rows × 12 cols -->
  <g fill="#b8b3a8">
    <circle cx="450" cy="245" r="3.5"/><circle cx="472" cy="245" r="3.5"/><circle cx="494" cy="245" r="3.5"/><circle cx="516" cy="245" r="3.5"/><circle cx="538" cy="245" r="3.5"/><circle cx="560" cy="245" r="3.5"/><circle cx="582" cy="245" r="3.5"/><circle cx="604" cy="245" r="3.5"/><circle cx="626" cy="245" r="3.5"/><circle cx="648" cy="245" r="3.5"/><circle cx="670" cy="245" r="3.5"/><circle cx="692" cy="245" r="3.5"/>
    <circle cx="450" cy="267" r="3.5"/><circle cx="472" cy="267" r="3.5"/><circle cx="494" cy="267" r="3.5"/><circle cx="516" cy="267" r="3.5"/><circle cx="538" cy="267" r="3.5"/><circle cx="560" cy="267" r="3.5"/><circle cx="582" cy="267" r="3.5"/><circle cx="604" cy="267" r="3.5"/><circle cx="626" cy="267" r="3.5"/><circle cx="648" cy="267" r="3.5"/><circle cx="670" cy="267" r="3.5"/><circle cx="692" cy="267" r="3.5"/>
    <circle cx="450" cy="289" r="3.5"/><circle cx="472" cy="289" r="3.5"/><circle cx="494" cy="289" r="3.5"/><circle cx="516" cy="289" r="3.5"/><circle cx="538" cy="289" r="3.5"/><circle cx="560" cy="289" r="3.5"/><circle cx="582" cy="289" r="3.5"/><circle cx="604" cy="289" r="3.5"/><circle cx="626" cy="289" r="3.5"/><circle cx="648" cy="289" r="3.5"/><circle cx="670" cy="289" r="3.5"/><circle cx="692" cy="289" r="3.5"/>
    <circle cx="450" cy="311" r="3.5"/><circle cx="472" cy="311" r="3.5"/><circle cx="494" cy="311" r="3.5"/><circle cx="516" cy="311" r="3.5"/><circle cx="538" cy="311" r="3.5"/><circle cx="560" cy="311" r="3.5"/><circle cx="582" cy="311" r="3.5"/><circle cx="604" cy="311" r="3.5"/><circle cx="626" cy="311" r="3.5"/><circle cx="648" cy="311" r="3.5"/><circle cx="670" cy="311" r="3.5"/><circle cx="692" cy="311" r="3.5"/>
    <circle cx="450" cy="333" r="3.5"/><circle cx="472" cy="333" r="3.5"/><circle cx="494" cy="333" r="3.5"/><circle cx="516" cy="333" r="3.5"/><circle cx="538" cy="333" r="3.5"/><circle cx="560" cy="333" r="3.5"/><circle cx="582" cy="333" r="3.5"/><circle cx="604" cy="333" r="3.5"/><circle cx="626" cy="333" r="3.5"/><circle cx="648" cy="333" r="3.5"/><circle cx="670" cy="333" r="3.5"/><circle cx="692" cy="333" r="3.5"/>
    <circle cx="450" cy="355" r="3.5"/><circle cx="472" cy="355" r="3.5"/><circle cx="494" cy="355" r="3.5"/><circle cx="516" cy="355" r="3.5"/><circle cx="538" cy="355" r="3.5"/><circle cx="560" cy="355" r="3.5"/><circle cx="582" cy="355" r="3.5"/><circle cx="604" cy="355" r="3.5"/><circle cx="626" cy="355" r="3.5"/><circle cx="648" cy="355" r="3.5"/><circle cx="670" cy="355" r="3.5"/><circle cx="692" cy="355" r="3.5"/>
  </g>
  <!-- relevant docs (teal) -->
  <g fill="#2a4747">
    <circle cx="494" cy="267" r="4.5"/>
    <circle cx="582" cy="311" r="4.5"/>
    <circle cx="648" cy="333" r="4.5"/>
    <circle cx="538" cy="355" r="4.5"/>
  </g>

  <!-- pool legend -->
  <g font-size="11" fill="#6b665e" font-style="italic">
    <circle cx="438" cy="436" r="3.5" fill="#b8b3a8"/>
    <text x="450" y="440">retrieved</text>
    <circle cx="512" cy="436" r="4.5" fill="#2a4747"/>
    <text x="522" y="440" fill="#2a4747">relevant</text>
  </g>

  <!-- Rerank -->
  <rect x="740" y="280" width="140" height="80" fill="#faf7f2" stroke="#1a1a1a" stroke-width="2" rx="4"/>
  <text x="810" y="313" font-size="15" fill="#1a1a1a" text-anchor="middle" font-weight="600">Rerank</text>
  <text x="810" y="335" font-size="12" fill="#6b665e" text-anchor="middle" font-style="italic">deep, slow</text>

  <!-- Generate -->
  <rect x="920" y="280" width="130" height="80" fill="#faf7f2" stroke="#1a1a1a" stroke-width="2" rx="4"/>
  <text x="985" y="313" font-size="15" fill="#1a1a1a" text-anchor="middle" font-weight="600">Generate</text>
  <text x="985" y="335" font-size="12" fill="#6b665e" text-anchor="middle" font-style="italic">LLM / answer</text>

  <!-- Flow arrows -->
  <line x1="180" y1="320" x2="234" y2="320" stroke="#1a1a1a" stroke-width="2" marker-end="url(#arrow)"/>
  <line x1="402" y1="320" x2="424" y2="320" stroke="#1a1a1a" stroke-width="2" marker-end="url(#arrow)"/>
  <line x1="712" y1="320" x2="734" y2="320" stroke="#1a1a1a" stroke-width="2" marker-end="url(#arrow)"/>
  <line x1="882" y1="320" x2="914" y2="320" stroke="#1a1a1a" stroke-width="2" marker-end="url(#arrow)"/>
</svg>
```

- [ ] **Step 2: Validate s1**

```bash
xmllint --noout docs/presentations/ragtune-diagrams/slide_rag_pipeline_s1.svg
```
Expected: no output (success). If errors, fix and re-run.

- [ ] **Step 3: Write slide_rag_pipeline_s2.svg**

Copy s1 exactly, then add these elements before the closing `</svg>` tag:

```svg
  <!-- s2: annotation labels -->
  <text x="700" y="470" font-size="12" fill="#6b665e" text-anchor="end" font-style="italic">top-K → rerank</text>
  <text x="570" y="200" font-size="13" fill="#1a1a1a" text-anchor="middle" font-weight="600">N = 1,000 candidates</text>
```

- [ ] **Step 4: Validate s2**

```bash
xmllint --noout docs/presentations/ragtune-diagrams/slide_rag_pipeline_s2.svg
```

- [ ] **Step 5: Write slide_rag_pipeline_s3.svg**

Copy s2 exactly, then add these elements before `</svg>`:

```svg
  <!-- s3: broken return path arc from Generate back toward Query -->
  <path d="M 985 280 C 985 155 130 155 130 270"
        fill="none" stroke="#b8b3a8" stroke-width="2" stroke-dasharray="6,4"/>
  <!-- ✗ mark at arc midpoint ~(555,152) -->
  <g stroke="#b8b3a8" stroke-width="2.5" stroke-linecap="round">
    <line x1="548" y1="145" x2="562" y2="159"/>
    <line x1="562" y1="145" x2="548" y2="159"/>
  </g>

  <!-- footer -->
  <line x1="80" y1="560" x2="200" y2="560" stroke="#c8553d" stroke-width="1.5"/>
  <text x="80" y="588" font-size="16" fill="#1a1a1a">The pipeline has no return path.</text>
  <text x="80" y="613" font-size="14" fill="#c8553d" font-weight="700">No feedback. Every query treated identically.</text>
```

- [ ] **Step 6: Validate s3**

```bash
xmllint --noout docs/presentations/ragtune-diagrams/slide_rag_pipeline_s3.svg
```

- [ ] **Step 7: Open all three in browser for visual check**

```bash
open docs/presentations/ragtune-diagrams/slide_rag_pipeline_s1.svg
open docs/presentations/ragtune-diagrams/slide_rag_pipeline_s2.svg
open docs/presentations/ragtune-diagrams/slide_rag_pipeline_s3.svg
```

Verify: title visible top-left, pipeline left-to-right, teal dots inside pool, s3 shows faint gray arc with ✗.

- [ ] **Step 8: Commit**

```bash
git add docs/presentations/ragtune-diagrams/slide_rag_pipeline_s*.svg
git commit -m "feat(figures): add slide_rag_pipeline step files (A1)"
```

---

## Task 2: A2 — slide_ragtune_loop (The feedback loop)

**Files:**
- Create: `docs/presentations/ragtune-diagrams/slide_ragtune_loop_s1.svg`
- Create: `docs/presentations/ragtune-diagrams/slide_ragtune_loop_s2.svg`
- Create: `docs/presentations/ragtune-diagrams/slide_ragtune_loop_s3.svg`

**Geometry:** 4 boxes in a clockwise square formation, centered on the canvas.

Box positions and centers:
```
CandidatePool (cool stroke #2a4747):  x=170,  y=270, w=160, h=80  → center (250, 310)
Estimator     (ink stroke):           x=770,  y=270, w=160, h=80  → center (850, 310)
Scheduler     (ink stroke):           x=770,  y=460, w=160, h=80  → center (850, 500)
Reranker      (ink stroke):           x=170,  y=460, w=160, h=80  → center (250, 500)
```

Clockwise arrows (s2):
```
Pool→Estimator:   (330,310) → (770,310)  horizontal rightward
Estimator→Scheduler: (850,350) → (850,460)  vertical downward
Scheduler→Reranker:  (770,500) → (330,500)  horizontal leftward
Reranker→Pool:    (250,460) → (250,350)  vertical upward
```

Arrow labels (s2, 13px italic muted, placed near midpoint of each arrow):
```
"priorities"  x=540, y=298  (above Pool→Estimator)
"batch"       x=870, y=410  (right of Estimator→Scheduler)
"scores"      x=540, y=518  (below Scheduler→Reranker)
"update"      x=145, y=410  (left of Reranker→Pool)
```

Feedback arrow (s3): dashed accent curve from Reranker to Estimator
```
<path d="M 330,500 C 500,590 620,200 770,310"
      fill="none" stroke="#c8553d" stroke-width="2.4"
      stroke-dasharray="8,5" marker-end="url(#arrowAccent)"/>
<text x="530" y="585" font-size="13" fill="#c8553d" font-style="italic"
      text-anchor="middle">signal</text>
```

CostTracker bar (s3):
```
<!-- label -->
<text x="900" y="148" font-size="12" fill="#6b665e" font-weight="600" letter-spacing="0.5">BUDGET</text>
<!-- track -->
<rect x="890" y="155" width="150" height="16" fill="none" stroke="#1a1a1a" stroke-width="1.2" rx="2"/>
<!-- fill (60% consumed) -->
<rect x="890" y="155" width="90" height="16" fill="#1a1a1a" fill-opacity="0.15" rx="2"/>
<text x="965" y="188" font-size="11" fill="#6b665e" text-anchor="middle" font-style="italic">60% consumed</text>
```

- [ ] **Step 1: Write slide_ragtune_loop_s1.svg**

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 720"
     font-family="Georgia, 'Times New Roman', serif" width="1100" height="720">
  <!-- slide_ragtune_loop s1: 4 component boxes, no arrows -->
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

  <rect width="1100" height="720" fill="#faf7f2"/>

  <!-- Header -->
  <text x="80" y="68" font-size="32" font-weight="400" fill="#1a1a1a" letter-spacing="0.2">RAGtune: The Loop</text>
  <text x="80" y="94" font-size="16" font-style="italic" fill="#6b665e" letter-spacing="0.3">Budget-aware iterative reranking</text>
  <line x1="80" y1="112" x2="200" y2="112" stroke="#1a1a1a" stroke-width="1.2"/>

  <!-- CandidatePool (cool teal border) -->
  <rect x="170" y="270" width="160" height="80" fill="#faf7f2" stroke="#2a4747" stroke-width="2.2" rx="4"/>
  <text x="250" y="303" font-size="14" fill="#2a4747" text-anchor="middle" font-weight="600">CandidatePool</text>
  <text x="250" y="323" font-size="11" fill="#6b665e" text-anchor="middle" font-style="italic">CANDIDATE → RERANKED</text>

  <!-- Estimator -->
  <rect x="770" y="270" width="160" height="80" fill="#faf7f2" stroke="#1a1a1a" stroke-width="2" rx="4"/>
  <text x="850" y="305" font-size="14" fill="#1a1a1a" text-anchor="middle" font-weight="600">Estimator</text>
  <text x="850" y="325" font-size="11" fill="#6b665e" text-anchor="middle" font-style="italic">value(pool)</text>

  <!-- Scheduler -->
  <rect x="770" y="460" width="160" height="80" fill="#faf7f2" stroke="#1a1a1a" stroke-width="2" rx="4"/>
  <text x="850" y="495" font-size="14" fill="#1a1a1a" text-anchor="middle" font-weight="600">Scheduler</text>
  <text x="850" y="515" font-size="11" fill="#6b665e" text-anchor="middle" font-style="italic">select_batch()</text>

  <!-- Reranker -->
  <rect x="170" y="460" width="160" height="80" fill="#faf7f2" stroke="#1a1a1a" stroke-width="2" rx="4"/>
  <text x="250" y="495" font-size="14" fill="#1a1a1a" text-anchor="middle" font-weight="600">Reranker</text>
  <text x="250" y="515" font-size="11" fill="#6b665e" text-anchor="middle" font-style="italic">rerank(batch)</text>
</svg>
```

- [ ] **Step 2: Validate s1**

```bash
xmllint --noout docs/presentations/ragtune-diagrams/slide_ragtune_loop_s1.svg
```

- [ ] **Step 3: Write slide_ragtune_loop_s2.svg**

Copy s1, add before `</svg>`:

```svg
  <!-- s2: clockwise forward-flow arrows + labels -->
  <!-- Pool → Estimator -->
  <line x1="330" y1="310" x2="770" y2="310" stroke="#1a1a1a" stroke-width="2" marker-end="url(#arrow)"/>
  <text x="550" y="298" font-size="13" fill="#6b665e" text-anchor="middle" font-style="italic">priorities</text>

  <!-- Estimator → Scheduler -->
  <line x1="850" y1="350" x2="850" y2="460" stroke="#1a1a1a" stroke-width="2" marker-end="url(#arrow)"/>
  <text x="876" y="410" font-size="13" fill="#6b665e" font-style="italic">batch</text>

  <!-- Scheduler → Reranker -->
  <line x1="770" y1="500" x2="330" y2="500" stroke="#1a1a1a" stroke-width="2" marker-end="url(#arrow)"/>
  <text x="550" y="518" font-size="13" fill="#6b665e" text-anchor="middle" font-style="italic">scores</text>

  <!-- Reranker → Pool -->
  <line x1="250" y1="460" x2="250" y2="350" stroke="#1a1a1a" stroke-width="2" marker-end="url(#arrow)"/>
  <text x="224" y="410" font-size="13" fill="#6b665e" text-anchor="end" font-style="italic">update</text>
```

- [ ] **Step 4: Validate s2**

```bash
xmllint --noout docs/presentations/ragtune-diagrams/slide_ragtune_loop_s2.svg
```

- [ ] **Step 5: Write slide_ragtune_loop_s3.svg**

Copy s2, add before `</svg>`:

```svg
  <!-- s3: dashed accent feedback arrow Reranker → Estimator -->
  <path d="M 330,500 C 500,590 620,200 770,310"
        fill="none" stroke="#c8553d" stroke-width="2.4"
        stroke-dasharray="8,5" marker-end="url(#arrowAccent)"/>
  <text x="530" y="582" font-size="13" fill="#c8553d" text-anchor="middle" font-style="italic">signal</text>

  <!-- CostTracker budget bar (top-right) -->
  <text x="900" y="148" font-size="11" fill="#6b665e" font-weight="600" letter-spacing="1">BUDGET</text>
  <rect x="890" y="155" width="150" height="16" fill="none" stroke="#1a1a1a" stroke-width="1.2" rx="2"/>
  <rect x="890" y="155" width="90"  height="16" fill="#1a1a1a" fill-opacity="0.15" rx="2"/>
  <text x="965" y="188" font-size="11" fill="#6b665e" text-anchor="middle" font-style="italic">60% consumed</text>

  <!-- footer -->
  <line x1="80" y1="620" x2="200" y2="620" stroke="#c8553d" stroke-width="1.5"/>
  <text x="80" y="648" font-size="16" fill="#1a1a1a">Same budget. Scheduling driven by feedback, not retrieval rank.</text>
  <text x="80" y="673" font-size="14" fill="#c8553d" font-weight="700">The feedback signal changes what gets scheduled next.</text>
```

- [ ] **Step 6: Validate s3 and visual check**

```bash
xmllint --noout docs/presentations/ragtune-diagrams/slide_ragtune_loop_s3.svg
open docs/presentations/ragtune-diagrams/slide_ragtune_loop_s1.svg
open docs/presentations/ragtune-diagrams/slide_ragtune_loop_s2.svg
open docs/presentations/ragtune-diagrams/slide_ragtune_loop_s3.svg
```

Verify: 4 boxes in square, s2 adds clockwise arrows with labels, s3 adds terracotta dashed feedback curve and budget bar.

- [ ] **Step 7: Commit**

```bash
git add docs/presentations/ragtune-diagrams/slide_ragtune_loop_s*.svg
git commit -m "feat(figures): add slide_ragtune_loop step files (A2)"
```

---

## Task 3: A3 — slide_estimator_matrix (Estimator concept)

**Files:**
- Create: `docs/presentations/ragtune-diagrams/slide_estimator_matrix_s1.svg`
- Create: `docs/presentations/ragtune-diagrams/slide_estimator_matrix_s2.svg`
- Create: `docs/presentations/ragtune-diagrams/slide_estimator_matrix_s3.svg`

**Geometry:**
```
Chart area: x_left=200, x_right=950, y_top=160, y_bottom=590
X-axis label: "feedback richness →"   x=575, y=640
Y-axis label: "← signal type"         rotated -90°, x=130, y=375

Estimator points:
  Baseline:  cx=260, cy=545  (low feedback, structural)
  Utility:   cx=380, cy=400  (low-mid feedback, mid signal)
  Similarity:cx=620, cy=250  (mid feedback, semantic)
  ReformIR:  cx=860, cy=195  (high feedback, semantic)

Diagonal sweet-spot band (s3):
  A parallelogram strip from (200,500) → (400,200) → (500,160) → (260,490) → close
  Use a rect rotated ~-35°, or a polygon with fill cool fill-opacity=0.08

Axis ticks and labels:
  X: "none" at x=200, y=615; "rich" at x=950, y=615
  Y: "structural" at x=185, y=590 text-anchor=end; "semantic" at x=185, y=165 text-anchor=end
```

- [ ] **Step 1: Write slide_estimator_matrix_s1.svg**

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 720"
     font-family="Georgia, 'Times New Roman', serif" width="1100" height="720">
  <!-- slide_estimator_matrix s1: axes only -->
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

  <rect width="1100" height="720" fill="#faf7f2"/>

  <!-- Header -->
  <text x="80" y="68" font-size="32" font-weight="400" fill="#1a1a1a" letter-spacing="0.2">Estimator: converting feedback into priorities</text>
  <text x="80" y="94" font-size="16" font-style="italic" fill="#6b665e" letter-spacing="0.3">Which estimator for which situation?</text>
  <line x1="80" y1="112" x2="200" y2="112" stroke="#1a1a1a" stroke-width="1.2"/>

  <!-- Grid lines -->
  <line x1="200" y1="160" x2="200" y2="590" stroke="#e8e2d6" stroke-width="1"/>
  <line x1="200" y1="590" x2="950" y2="590" stroke="#e8e2d6" stroke-width="1"/>
  <line x1="575" y1="160" x2="575" y2="590" stroke="#e8e2d6" stroke-width="0.8" stroke-dasharray="4,4"/>
  <line x1="200" y1="375" x2="950" y2="375" stroke="#e8e2d6" stroke-width="0.8" stroke-dasharray="4,4"/>

  <!-- Axes -->
  <line x1="200" y1="590" x2="950" y2="590" stroke="#1a1a1a" stroke-width="1.8" marker-end="url(#arrow)"/>
  <line x1="200" y1="590" x2="200" y2="155" stroke="#1a1a1a" stroke-width="1.8" marker-end="url(#arrow)"/>

  <!-- Axis labels -->
  <text x="575" y="635" font-size="14" fill="#6b665e" text-anchor="middle" font-style="italic">feedback richness →</text>
  <text x="130" y="375" font-size="14" fill="#6b665e" text-anchor="middle" font-style="italic"
        transform="rotate(-90, 130, 375)">signal type →</text>

  <!-- Axis end labels -->
  <text x="200" y="615" font-size="12" fill="#8a857a" text-anchor="middle">none</text>
  <text x="950" y="615" font-size="12" fill="#8a857a" text-anchor="middle">rich</text>
  <text x="185" y="594" font-size="12" fill="#8a857a" text-anchor="end">structural</text>
  <text x="185" y="168" font-size="12" fill="#8a857a" text-anchor="end">semantic</text>
</svg>
```

- [ ] **Step 2: Validate s1**

```bash
xmllint --noout docs/presentations/ragtune-diagrams/slide_estimator_matrix_s1.svg
```

- [ ] **Step 3: Write slide_estimator_matrix_s2.svg**

Copy s1, add before `</svg>`:

```svg
  <!-- s2: 4 estimator points -->
  <!-- Baseline -->
  <circle cx="260" cy="545" r="8" fill="#b8b3a8"/>
  <text x="260" y="535" font-size="13" fill="#8a857a" text-anchor="middle" font-weight="600">Baseline</text>
  <text x="260" y="570" font-size="11" fill="#8a857a" text-anchor="middle" font-style="italic">retrieval score only</text>

  <!-- Utility -->
  <circle cx="400" cy="400" r="8" fill="#6b665e"/>
  <text x="400" y="388" font-size="13" fill="#6b665e" text-anchor="middle" font-weight="600">Utility</text>
  <text x="400" y="422" font-size="11" fill="#6b665e" text-anchor="middle" font-style="italic">metadata overlap</text>

  <!-- Similarity -->
  <circle cx="630" cy="255" r="8" fill="#1a1a1a"/>
  <text x="630" y="243" font-size="13" fill="#1a1a1a" text-anchor="middle" font-weight="600">Similarity</text>
  <text x="630" y="275" font-size="11" fill="#6b665e" text-anchor="middle" font-style="italic">cosine to winners</text>

  <!-- ReformIR -->
  <circle cx="860" cy="200" r="9" fill="#2a4747"/>
  <text x="860" y="188" font-size="13" fill="#2a4747" text-anchor="middle" font-weight="600">ReformIR</text>
  <text x="860" y="222" font-size="11" fill="#6b665e" text-anchor="middle" font-style="italic">regression: source → score</text>
```

- [ ] **Step 4: Validate s2**

```bash
xmllint --noout docs/presentations/ragtune-diagrams/slide_estimator_matrix_s2.svg
```

- [ ] **Step 5: Write slide_estimator_matrix_s3.svg**

Copy s2, add before `</svg>`:

```svg
  <!-- s3: diagonal sweet-spot band (bottom-left to top-right) -->
  <polygon points="200,540 350,170 500,160 320,560"
           fill="#2a4747" fill-opacity="0.07" stroke="none"/>
  <text x="285" y="350" font-size="12" fill="#2a4747" font-style="italic"
        transform="rotate(-50, 285, 350)">sweet spot</text>

  <!-- accent circle around ReformIR -->
  <circle cx="860" cy="200" r="22" fill="none" stroke="#c8553d" stroke-width="1.8" stroke-dasharray="4,3"/>

  <!-- footer -->
  <line x1="80" y1="638" x2="200" y2="638" stroke="#c8553d" stroke-width="1.5"/>
  <text x="80" y="664" font-size="15" fill="#1a1a1a">ReformIR: after 3 scored docs, source weights update every iteration.</text>
  <text x="80" y="688" font-size="14" fill="#c8553d" font-weight="700">Use as much feedback as you have.</text>
```

- [ ] **Step 6: Validate s3 and visual check**

```bash
xmllint --noout docs/presentations/ragtune-diagrams/slide_estimator_matrix_s3.svg
open docs/presentations/ragtune-diagrams/slide_estimator_matrix_s1.svg
open docs/presentations/ragtune-diagrams/slide_estimator_matrix_s2.svg
open docs/presentations/ragtune-diagrams/slide_estimator_matrix_s3.svg
```

Verify: axes with labels, 4 points placed bottom-left to top-right, s3 shows diagonal teal band and accent ring around ReformIR.

- [ ] **Step 7: Commit**

```bash
git add docs/presentations/ragtune-diagrams/slide_estimator_matrix_s*.svg
git commit -m "feat(figures): add slide_estimator_matrix step files (A3)"
```

---

## Task 4: A4 — slide_results_pareto (Pareto bar chart)

**Files:**
- Create: `docs/presentations/ragtune-diagrams/slide_results_pareto_s1.svg`
- Create: `docs/presentations/ragtune-diagrams/slide_results_pareto_s2.svg`
- Create: `docs/presentations/ragtune-diagrams/slide_results_pareto_s3.svg`

**Geometry:**
```
Chart area: x_axis from x=200 to x=920, y_base=570 (bottom), y_top=170 (top)
Chart height: 400px

Y-axis: NDCG@5 from 0.60 to 0.78, range=0.18
scale_y = 400 / 0.18 = 2222 px per NDCG unit

X-axis: avg rerank docs (budget), 0–32 docs
scale_x = 720 / 32 = 22.5 px per doc
x_origin = 200 (0 docs)

Bar positions (center_x = x_origin + docs * scale_x):
  BM25      (0  docs): cx=200,  bar_x=175, bar_w=50
  tight     (5  docs): cx=313,  bar_x=288, bar_w=50
  feedback  (10 docs): cx=425,  bar_x=400, bar_w=50  ← accent in s3
  medium    (15 docs): cx=538,  bar_x=513, bar_w=50
  loose     (30 docs): cx=875,  bar_x=850, bar_w=50

Bar heights (y_top = y_base - (value - 0.60) * scale_y):
  BM25     (0.656): bar_top = 570 - 0.056*2222 = 570 - 124 = 446
  tight    (0.720): bar_top = 570 - 0.120*2222 = 570 - 267 = 303
  feedback (0.760): bar_top = 570 - 0.160*2222 = 570 - 356 = 214  ← accent
  medium   (0.749): bar_top = 570 - 0.149*2222 = 570 - 331 = 239
  loose    (0.748): bar_top = 570 - 0.148*2222 = 570 - 329 = 241

Y-axis ticks at NDCG 0.60, 0.65, 0.70, 0.75:
  0.60: y=570  0.65: y=459  0.70: y=348  0.75: y=237

X-axis ticks: 0→x=200, 5→x=313, 10→x=425, 15→x=538, 20→x=650, 25→x=763, 30→x=875
```

- [ ] **Step 1: Write slide_results_pareto_s1.svg**

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 720"
     font-family="Georgia, 'Times New Roman', serif" width="1100" height="720">
  <!-- slide_results_pareto s1: axes only -->
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

  <rect width="1100" height="720" fill="#faf7f2"/>

  <!-- Header -->
  <text x="80" y="68" font-size="32" font-weight="400" fill="#1a1a1a" letter-spacing="0.2">Results: the Pareto picture</text>
  <text x="80" y="94" font-size="16" font-style="italic" fill="#6b665e" letter-spacing="0.3">3 datasets · MonoT5 reranker · avg NDCG@5</text>
  <line x1="80" y1="112" x2="200" y2="112" stroke="#1a1a1a" stroke-width="1.2"/>

  <!-- Chart grid lines -->
  <line x1="200" y1="459" x2="920" y2="459" stroke="#e8e2d6" stroke-width="0.8"/>
  <line x1="200" y1="348" x2="920" y2="348" stroke="#e8e2d6" stroke-width="0.8"/>
  <line x1="200" y1="237" x2="920" y2="237" stroke="#e8e2d6" stroke-width="0.8"/>

  <!-- Axes -->
  <line x1="200" y1="570" x2="920" y2="570" stroke="#1a1a1a" stroke-width="1.8" marker-end="url(#arrow)"/>
  <line x1="200" y1="570" x2="200" y2="165" stroke="#1a1a1a" stroke-width="1.8" marker-end="url(#arrow)"/>

  <!-- Y-axis tick labels -->
  <text x="188" y="574" font-size="12" fill="#8a857a" text-anchor="end">0.60</text>
  <text x="188" y="463" font-size="12" fill="#8a857a" text-anchor="end">0.65</text>
  <text x="188" y="352" font-size="12" fill="#8a857a" text-anchor="end">0.70</text>
  <text x="188" y="241" font-size="12" fill="#8a857a" text-anchor="end">0.75</text>

  <!-- Y-axis label -->
  <text x="140" y="370" font-size="14" fill="#6b665e" text-anchor="middle"
        font-style="italic" transform="rotate(-90, 140, 370)">avg NDCG@5</text>

  <!-- X-axis tick labels -->
  <text x="200" y="592" font-size="12" fill="#8a857a" text-anchor="middle">0</text>
  <text x="313" y="592" font-size="12" fill="#8a857a" text-anchor="middle">5</text>
  <text x="425" y="592" font-size="12" fill="#8a857a" text-anchor="middle">10</text>
  <text x="538" y="592" font-size="12" fill="#8a857a" text-anchor="middle">15</text>
  <text x="875" y="592" font-size="12" fill="#8a857a" text-anchor="middle">30</text>

  <!-- X-axis label -->
  <text x="560" y="625" font-size="14" fill="#6b665e" text-anchor="middle" font-style="italic">avg rerank docs (budget)</text>
</svg>
```

- [ ] **Step 2: Validate s1**

```bash
xmllint --noout docs/presentations/ragtune-diagrams/slide_results_pareto_s1.svg
```

- [ ] **Step 3: Write slide_results_pareto_s2.svg**

Copy s1, add before `</svg>`:

```svg
  <!-- s2: gray bars — BM25, tight, medium, loose -->
  <!-- BM25 (0 docs, 0.656): bar_top=446, height=124 -->
  <rect x="175" y="446" width="50" height="124" fill="#b8b3a8" rx="2"/>
  <text x="200" y="440" font-size="12" fill="#8a857a" text-anchor="middle">BM25</text>
  <text x="200" y="428" font-size="11" fill="#8a857a" text-anchor="middle">0.656</text>

  <!-- tight/5 (5 docs, 0.720): bar_top=303, height=267 -->
  <rect x="288" y="303" width="50" height="267" fill="#b8b3a8" rx="2"/>
  <text x="313" y="297" font-size="12" fill="#8a857a" text-anchor="middle">tight/5</text>
  <text x="313" y="285" font-size="11" fill="#8a857a" text-anchor="middle">0.720</text>

  <!-- medium/15 (15 docs, 0.749): bar_top=239, height=331 -->
  <rect x="513" y="239" width="50" height="331" fill="#b8b3a8" rx="2"/>
  <text x="538" y="233" font-size="12" fill="#8a857a" text-anchor="middle">medium/15</text>
  <text x="538" y="221" font-size="11" fill="#8a857a" text-anchor="middle">0.749</text>

  <!-- loose/30 (30 docs, 0.748): bar_top=241, height=329 -->
  <rect x="850" y="241" width="50" height="329" fill="#b8b3a8" rx="2"/>
  <text x="875" y="235" font-size="12" fill="#8a857a" text-anchor="middle">loose/30</text>
  <text x="875" y="223" font-size="11" fill="#8a857a" text-anchor="middle">0.748</text>
```

- [ ] **Step 4: Validate s2**

```bash
xmllint --noout docs/presentations/ragtune-diagrams/slide_results_pareto_s2.svg
```

- [ ] **Step 5: Write slide_results_pareto_s3.svg**

Copy s2, add before `</svg>`:

```svg
  <!-- s3: accent bar — convergence feedback (10 docs, 0.760): bar_top=214, height=356 -->
  <rect x="400" y="214" width="50" height="356" fill="#c8553d" rx="2"/>
  <text x="425" y="208" font-size="13" fill="#c8553d" text-anchor="middle" font-weight="600">feedback</text>
  <text x="425" y="193" font-size="12" fill="#c8553d" text-anchor="middle" font-weight="700">0.760</text>

  <!-- delta annotation: bracket from loose bar top (241) to feedback bar top (214) -->
  <line x1="870" y1="241" x2="460" y2="214" stroke="#c8553d" stroke-width="1.2" stroke-dasharray="4,3"/>
  <text x="670" y="200" font-size="14" fill="#c8553d" text-anchor="middle" font-weight="700">+4% NDCG, ⅓ the budget</text>

  <!-- footer -->
  <line x1="80" y1="648" x2="200" y2="648" stroke="#c8553d" stroke-width="1.5"/>
  <text x="80" y="674" font-size="16" fill="#1a1a1a">More docs ≠ better results. Smarter scheduling does.</text>
  <text x="80" y="698" font-size="14" fill="#c8553d" font-weight="700">The ceiling isn't the budget — it's how you use it.</text>
```

*Note: all NDCG values are placeholders. To use real paper data, edit the `value` and recompute `bar_top = 570 - (value - 0.60) * 2222`, `height = 570 - bar_top`.*

- [ ] **Step 6: Validate s3 and visual check**

```bash
xmllint --noout docs/presentations/ragtune-diagrams/slide_results_pareto_s3.svg
open docs/presentations/ragtune-diagrams/slide_results_pareto_s1.svg
open docs/presentations/ragtune-diagrams/slide_results_pareto_s2.svg
open docs/presentations/ragtune-diagrams/slide_results_pareto_s3.svg
```

Verify: axes + grid, s2 adds 4 gray bars (medium ≈ loose height), s3 adds prominent accent bar between tight and medium, delta annotation visible.

- [ ] **Step 7: Commit**

```bash
git add docs/presentations/ragtune-diagrams/slide_results_pareto_s*.svg
git commit -m "feat(figures): add slide_results_pareto step files (A4)"
```

---

## Task 5: B1 — slide_ragtune_mapping (Science → software)

**Files:**
- Create: `talks/booking-2026/booking-diagrams/slide_ragtune_mapping_s1.svg`
- Create: `talks/booking-2026/booking-diagrams/slide_ragtune_mapping_s2.svg`
- Create: `talks/booking-2026/booking-diagrams/slide_ragtune_mapping_s3.svg`

**Geometry:**
```
Science boxes (top row, cool teal stroke, y=220, h=65):
  QUAM:     x=55,  w=130, cx=120, cy=252
  SUNAR:    x=225, w=130, cx=290, cy=252
  ORE:      x=395, w=130, cx=460, cy=252
  ReformIR: x=565, w=150, cx=640, cy=252
  CASE:     x=755, w=130, cx=820, cy=252

RAGtune loop bounding box (s2): x=100, y=415, w=880, h=110, rx=4
  stroke: #1a1a1a, stroke-dasharray="6,4", fill none
  label: "RAGtune Controller Loop" at x=540, y=408, 13px bold, muted, text-anchor=middle

Component slot boxes inside loop (s2, y=435, h=70, ink stroke):
  Estimator:     x=120, w=150, cx=195, cy=470
  Scheduler:     x=320, w=150, cx=395, cy=470
  Reranker:      x=520, w=150, cx=595, cy=470
  CandidatePool: x=730, w=180, cx=820, cy=470

Connection arrows (s3), dashed accent:
  QUAM (120,285)    → Estimator (195,435):   short near-vertical
  SUNAR (290,285)   → loop box center (540,415): curves to bounding box
  ORE (460,285)     → Scheduler (395,435):   near-vertical
  ReformIR (640,285)→ Estimator (195,435):   long left-diagonal
  CASE (820,285)    → Estimator (195,435):   long far-left diagonal

Label each arrow with the algorithm name at its midpoint.
For SUNAR: arrow tip touches the top edge of the loop bounding box at x=540.
```

- [ ] **Step 1: Write slide_ragtune_mapping_s1.svg**

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 720"
     font-family="Georgia, 'Times New Roman', serif" width="1100" height="720">
  <!-- slide_ragtune_mapping s1: 5 science system boxes -->
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

  <rect width="1100" height="720" fill="#faf7f2"/>

  <!-- Header -->
  <text x="80" y="68" font-size="32" font-weight="400" fill="#1a1a1a" letter-spacing="0.2">RAGtune: from science to software</text>
  <text x="80" y="94" font-size="16" font-style="italic" fill="#6b665e" letter-spacing="0.3">Five algorithms. One loop.</text>
  <line x1="80" y1="112" x2="200" y2="112" stroke="#1a1a1a" stroke-width="1.2"/>

  <!-- Science boxes (teal) -->
  <rect x="55"  y="220" width="130" height="65" fill="#faf7f2" stroke="#2a4747" stroke-width="2" rx="4"/>
  <text x="120" y="256" font-size="16" fill="#2a4747" text-anchor="middle" font-weight="600">QUAM</text>

  <rect x="225" y="220" width="130" height="65" fill="#faf7f2" stroke="#2a4747" stroke-width="2" rx="4"/>
  <text x="290" y="256" font-size="16" fill="#2a4747" text-anchor="middle" font-weight="600">SUNAR</text>

  <rect x="395" y="220" width="130" height="65" fill="#faf7f2" stroke="#2a4747" stroke-width="2" rx="4"/>
  <text x="460" y="256" font-size="16" fill="#2a4747" text-anchor="middle" font-weight="600">ORE</text>

  <rect x="565" y="220" width="150" height="65" fill="#faf7f2" stroke="#2a4747" stroke-width="2" rx="4"/>
  <text x="640" y="256" font-size="16" fill="#2a4747" text-anchor="middle" font-weight="600">ReformIR</text>

  <rect x="755" y="220" width="130" height="65" fill="#faf7f2" stroke="#2a4747" stroke-width="2" rx="4"/>
  <text x="820" y="256" font-size="16" fill="#2a4747" text-anchor="middle" font-weight="600">CASE</text>
</svg>
```

- [ ] **Step 2: Validate s1**

```bash
xmllint --noout /Users/avishekanand/talks/booking-2026/booking-diagrams/slide_ragtune_mapping_s1.svg
```

- [ ] **Step 3: Write slide_ragtune_mapping_s2.svg**

Copy s1, add before `</svg>`:

```svg
  <!-- s2: RAGtune controller loop with component slots -->
  <!-- bounding box -->
  <rect x="100" y="415" width="880" height="110" fill="none" stroke="#1a1a1a"
        stroke-width="1.5" stroke-dasharray="6,4" rx="4"/>
  <text x="540" y="408" font-size="13" fill="#6b665e" text-anchor="middle"
        font-weight="600" letter-spacing="0.5">RAGtune Controller Loop</text>

  <!-- Estimator slot -->
  <rect x="120" y="435" width="150" height="70" fill="#faf7f2" stroke="#1a1a1a" stroke-width="1.8" rx="4"/>
  <text x="195" y="467" font-size="14" fill="#1a1a1a" text-anchor="middle" font-weight="600">Estimator</text>
  <text x="195" y="487" font-size="11" fill="#6b665e" text-anchor="middle" font-style="italic">value(pool)</text>

  <!-- Scheduler slot -->
  <rect x="320" y="435" width="150" height="70" fill="#faf7f2" stroke="#1a1a1a" stroke-width="1.8" rx="4"/>
  <text x="395" y="467" font-size="14" fill="#1a1a1a" text-anchor="middle" font-weight="600">Scheduler</text>
  <text x="395" y="487" font-size="11" fill="#6b665e" text-anchor="middle" font-style="italic">select_batch()</text>

  <!-- Reranker slot -->
  <rect x="520" y="435" width="150" height="70" fill="#faf7f2" stroke="#1a1a1a" stroke-width="1.8" rx="4"/>
  <text x="595" y="467" font-size="14" fill="#1a1a1a" text-anchor="middle" font-weight="600">Reranker</text>
  <text x="595" y="487" font-size="11" fill="#6b665e" text-anchor="middle" font-style="italic">rerank(batch)</text>

  <!-- CandidatePool slot -->
  <rect x="730" y="435" width="180" height="70" fill="#faf7f2" stroke="#1a1a1a" stroke-width="1.8" rx="4"/>
  <text x="820" y="467" font-size="14" fill="#1a1a1a" text-anchor="middle" font-weight="600">CandidatePool</text>
  <text x="820" y="487" font-size="11" fill="#6b665e" text-anchor="middle" font-style="italic">CANDIDATE→RERANKED</text>
```

- [ ] **Step 4: Validate s2**

```bash
xmllint --noout /Users/avishekanand/talks/booking-2026/booking-diagrams/slide_ragtune_mapping_s2.svg
```

- [ ] **Step 5: Write slide_ragtune_mapping_s3.svg**

Copy s2, add before `</svg>`:

```svg
  <!-- s3: dashed accent connection arrows — each algorithm to its slot -->

  <!-- QUAM → Estimator (near-vertical) -->
  <line x1="120" y1="285" x2="175" y2="435"
        stroke="#c8553d" stroke-width="1.8" stroke-dasharray="7,4" marker-end="url(#arrowAccent)"/>

  <!-- SUNAR → loop bounding box top center (full loop) -->
  <line x1="290" y1="285" x2="400" y2="415"
        stroke="#c8553d" stroke-width="1.8" stroke-dasharray="7,4" marker-end="url(#arrowAccent)"/>
  <text x="330" y="360" font-size="11" fill="#c8553d" font-style="italic" text-anchor="middle">full loop</text>

  <!-- ORE → Scheduler (near-vertical) -->
  <line x1="460" y1="285" x2="420" y2="435"
        stroke="#c8553d" stroke-width="1.8" stroke-dasharray="7,4" marker-end="url(#arrowAccent)"/>

  <!-- ReformIR → Estimator (long left diagonal) -->
  <path d="M 640,285 C 580,360 350,390 220,435"
        fill="none" stroke="#c8553d" stroke-width="1.8" stroke-dasharray="7,4" marker-end="url(#arrowAccent)"/>

  <!-- CASE → Estimator (far-left diagonal) -->
  <path d="M 820,285 C 700,360 380,390 220,435"
        fill="none" stroke="#c8553d" stroke-width="1.8" stroke-dasharray="7,4" marker-end="url(#arrowAccent)"/>

  <!-- footer -->
  <line x1="80" y1="590" x2="200" y2="590" stroke="#c8553d" stroke-width="1.5"/>
  <text x="80" y="618" font-size="16" fill="#1a1a1a">Each algorithm plugs into the component it replaces.</text>
  <text x="80" y="643" font-size="14" fill="#c8553d" font-weight="700">RAGtune is the software that makes the science runnable.</text>
```

- [ ] **Step 6: Validate s3 and visual check**

```bash
xmllint --noout /Users/avishekanand/talks/booking-2026/booking-diagrams/slide_ragtune_mapping_s3.svg
open /Users/avishekanand/talks/booking-2026/booking-diagrams/slide_ragtune_mapping_s1.svg
open /Users/avishekanand/talks/booking-2026/booking-diagrams/slide_ragtune_mapping_s2.svg
open /Users/avishekanand/talks/booking-2026/booking-diagrams/slide_ragtune_mapping_s3.svg
```

Verify: 5 teal science boxes top row; s2 adds dashed loop box with 4 slots below; s3 adds accent arrows connecting each algorithm to its slot (SUNAR to bounding box center, ReformIR+CASE curve left to Estimator).

- [ ] **Step 7: Commit**

```bash
cd /Users/avishekanand/talks/booking-2026
git init 2>/dev/null || true
git add booking-diagrams/slide_ragtune_mapping_s*.svg
git commit -m "feat(figures): add slide_ragtune_mapping step files (B1)"
```

---

## Task 6: B2 — slide_ragtune_status ("We're cooking" callout)

**Files:**
- Create: `talks/booking-2026/booking-diagrams/slide_ragtune_status_s1.svg`
- Create: `talks/booking-2026/booking-diagrams/slide_ragtune_status_s2.svg`
- Create: `talks/booking-2026/booking-diagrams/slide_ragtune_status_s3.svg`

**Geometry:**
```
s1: Typographic, centered
  "RAGtune" — x=550, y=310, font-size=64, ink, text-anchor=middle
  "· active development" — x=550, y=365, font-size=22, muted italic, text-anchor=middle
  "github.com/avishekanand/sir" — x=550, y=410, font-size=18, cool, text-anchor=middle

s2: Two columns added below
  Divider line: x1=200, y1=455, x2=900, y2=455, grid color
  Left column header "✓ running now" — x=320, y=490, cool, 16px bold, text-anchor=middle
  Left items (13px, cool, y starts 520 spacing 30):
    Controller loop, CandidatePool, Budget enforcement, ReformIR estimator, ActiveLearning scheduler
  Right column header "⟳ in progress" — x=730, y=490, muted, 16px bold, text-anchor=middle
  Right items (13px, muted, y starts 520 spacing 30):
    QUAM adapter, ORE bandit scheduler, CASE estimator, SUNAR loop

s3: Footer accent line + closing line
```

- [ ] **Step 1: Write slide_ragtune_status_s1.svg**

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 720"
     font-family="Georgia, 'Times New Roman', serif" width="1100" height="720">
  <!-- slide_ragtune_status s1: typographic title -->
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

  <rect width="1100" height="720" fill="#faf7f2"/>

  <!-- Header -->
  <text x="80" y="68" font-size="32" font-weight="400" fill="#1a1a1a" letter-spacing="0.2">RAGtune</text>
  <text x="80" y="94" font-size="16" font-style="italic" fill="#6b665e" letter-spacing="0.3">Turning the science into running software</text>
  <line x1="80" y1="112" x2="200" y2="112" stroke="#1a1a1a" stroke-width="1.2"/>

  <!-- Large centered title -->
  <text x="550" y="310" font-size="64" font-weight="400" fill="#1a1a1a"
        text-anchor="middle" letter-spacing="0.5">RAGtune</text>
  <text x="550" y="362" font-size="22" fill="#6b665e" text-anchor="middle"
        font-style="italic">· active development</text>
  <text x="550" y="408" font-size="18" fill="#2a4747" text-anchor="middle">github.com/avishekanand/sir</text>
</svg>
```

- [ ] **Step 2: Validate s1**

```bash
xmllint --noout /Users/avishekanand/talks/booking-2026/booking-diagrams/slide_ragtune_status_s1.svg
```

- [ ] **Step 3: Write slide_ragtune_status_s2.svg**

Copy s1, add before `</svg>`:

```svg
  <!-- s2: two-column checklist -->
  <line x1="200" y1="450" x2="900" y2="450" stroke="#e8e2d6" stroke-width="1"/>
  <line x1="545" y1="455" x2="545" y2="670" stroke="#e8e2d6" stroke-width="1"/>

  <!-- Left: running now -->
  <text x="320" y="485" font-size="15" fill="#2a4747" text-anchor="middle" font-weight="700">✓  running now</text>
  <text x="250" y="518" font-size="13" fill="#2a4747">Controller loop</text>
  <text x="250" y="548" font-size="13" fill="#2a4747">CandidatePool state machine</text>
  <text x="250" y="578" font-size="13" fill="#2a4747">Budget enforcement (4 constraints)</text>
  <text x="250" y="608" font-size="13" fill="#2a4747">ReformIR estimator</text>
  <text x="250" y="638" font-size="13" fill="#2a4747">ActiveLearning scheduler</text>

  <!-- Right: in progress -->
  <text x="730" y="485" font-size="15" fill="#6b665e" text-anchor="middle" font-weight="700">⟳  in progress</text>
  <text x="565" y="518" font-size="13" fill="#6b665e">QUAM adapter</text>
  <text x="565" y="548" font-size="13" fill="#6b665e">ORE bandit scheduler</text>
  <text x="565" y="578" font-size="13" fill="#6b665e">CASE estimator</text>
  <text x="565" y="608" font-size="13" fill="#6b665e">SUNAR loop integration</text>
```

- [ ] **Step 4: Validate s2**

```bash
xmllint --noout /Users/avishekanand/talks/booking-2026/booking-diagrams/slide_ragtune_status_s2.svg
```

- [ ] **Step 5: Write slide_ragtune_status_s3.svg**

Copy s2, add before `</svg>`:

```svg
  <!-- s3: footer closing line -->
  <line x1="80" y1="680" x2="200" y2="680" stroke="#c8553d" stroke-width="1.5"/>
  <text x="80" y="700" font-size="14" fill="#c8553d" font-weight="700">The science is done. The software is being built.</text>
```

- [ ] **Step 6: Validate s3 and visual check**

```bash
xmllint --noout /Users/avishekanand/talks/booking-2026/booking-diagrams/slide_ragtune_status_s3.svg
open /Users/avishekanand/talks/booking-2026/booking-diagrams/slide_ragtune_status_s1.svg
open /Users/avishekanand/talks/booking-2026/booking-diagrams/slide_ragtune_status_s2.svg
open /Users/avishekanand/talks/booking-2026/booking-diagrams/slide_ragtune_status_s3.svg
```

Verify: s1 centered large "RAGtune" title; s2 adds two-column checklist (teal left, muted right); s3 adds terracotta closing line at bottom.

- [ ] **Step 7: Commit**

```bash
cd /Users/avishekanand/talks/booking-2026
git add booking-diagrams/slide_ragtune_status_s*.svg
git commit -m "feat(figures): add slide_ragtune_status step files (B2)"
```

---

## Task 7: Marp integration — RAGtune standalone deck

**Files:**
- Modify: `docs/presentations/ragtune-feedback-driven-retrieval-group-talk.md`

Replace each of the 4 text slides with a 3-step full-bleed SVG sequence. The text content of the slide becomes the SVG header/footer — remove the text slide entirely and replace with the step sequence.

**Slides to replace** (identified by their `#` heading):

1. `# Standard RAG is a one-way street` → `slide_rag_pipeline_s1/s2/s3`
2. `# RAGtune: The Loop` → `slide_ragtune_loop_s1/s2/s3`
3. `# Estimator: converting feedback into priorities` → `slide_estimator_matrix_s1/s2/s3`
4. `# Results: the Pareto picture` → `slide_results_pareto_s1/s2/s3`

**Update the deck's frontmatter** to add the shared visual system style block. Replace the existing `style:` block with:

```yaml
style: |
  :root {
    --paper: #faf7f2;
    --ink:   #1a1a1a;
    --muted: #6b665e;
    --accent: #c8553d;
    --cool:  #2a4747;
  }
  section {
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 28px;
    background: var(--paper);
    color: var(--ink);
    padding: 70px 90px;
  }
  h1 { color: var(--ink); border-bottom: 2px solid var(--accent); padding-bottom: 8px; }
  h2 { color: var(--ink); }
  code { background: #f0ede8; padding: 2px 6px; border-radius: 4px; }
  pre  { background: #1e1e1e; color: #d4d4d4; border-radius: 8px; }
  table { font-size: 22px; }
```

- [ ] **Step 1: Back up the existing deck**

```bash
cp docs/presentations/ragtune-feedback-driven-retrieval-group-talk.md \
   docs/presentations/ragtune-feedback-driven-retrieval-group-talk.md.bak
```

- [ ] **Step 2: Update frontmatter style block**

Open `docs/presentations/ragtune-feedback-driven-retrieval-group-talk.md`. Replace the `style: |` block (lines 5–16 in the current file, from `style: |` through the closing `---`) with the style block above. Keep all other frontmatter fields unchanged (`marp: true`, `theme: default`, `paginate: true`).

- [ ] **Step 3: Replace slide A1 — "Standard RAG is a one-way street"**

Find the slide starting with `# Standard RAG is a one-way street` (currently ends at `> If you have budget for 10 reranks, you pick the top-10 by retrieval score. Done.`). Replace the entire slide (from `---` before the heading to `---` after the last line) with:

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

- [ ] **Step 4: Replace slide A2 — "RAGtune: The Loop"**

Find and replace the slide starting with `# RAGtune: The Loop` (the one with the Python `while budget remains:` code block):

```markdown
---
<!-- _header: '' -->
<!-- _footer: '' -->

![bg fit](ragtune-diagrams/slide_ragtune_loop_s1.svg)

---
<!-- _header: '' -->
<!-- _footer: '' -->

![bg fit](ragtune-diagrams/slide_ragtune_loop_s2.svg)

---
<!-- _header: '' -->
<!-- _footer: '' -->

![bg fit](ragtune-diagrams/slide_ragtune_loop_s3.svg)
```

- [ ] **Step 5: Replace slide A3 — "Estimator: converting feedback into priorities"**

Find and replace the slide starting with `# Estimator: converting feedback into priorities`:

```markdown
---
<!-- _header: '' -->
<!-- _footer: '' -->

![bg fit](ragtune-diagrams/slide_estimator_matrix_s1.svg)

---
<!-- _header: '' -->
<!-- _footer: '' -->

![bg fit](ragtune-diagrams/slide_estimator_matrix_s2.svg)

---
<!-- _header: '' -->
<!-- _footer: '' -->

![bg fit](ragtune-diagrams/slide_estimator_matrix_s3.svg)
```

- [ ] **Step 6: Replace slide A4 — "Results: the Pareto picture"**

Find and replace the slide starting with `# Results: the Pareto picture`:

```markdown
---
<!-- _header: '' -->
<!-- _footer: '' -->

![bg fit](ragtune-diagrams/slide_results_pareto_s1.svg)

---
<!-- _header: '' -->
<!-- _footer: '' -->

![bg fit](ragtune-diagrams/slide_results_pareto_s2.svg)

---
<!-- _header: '' -->
<!-- _footer: '' -->

![bg fit](ragtune-diagrams/slide_results_pareto_s3.svg)
```

- [ ] **Step 7: Verify slide count and no broken references**

```bash
grep -c "^---$" docs/presentations/ragtune-feedback-driven-retrieval-group-talk.md
# original had 22 slides; each replaced slide becomes 3 → net +8, expect ~30
grep "ragtune-diagrams/" docs/presentations/ragtune-feedback-driven-retrieval-group-talk.md | grep -v "_s[1-3]\.svg"
# expected: no output (all references use step files)
```

- [ ] **Step 8: Commit**

```bash
rm docs/presentations/ragtune-feedback-driven-retrieval-group-talk.md.bak
git add docs/presentations/ragtune-feedback-driven-retrieval-group-talk.md
git commit -m "feat: integrate SVG step figures into RAGtune marp deck"
```

---

## Task 8: Create booking-ragtune.md (Booking talk section)

**Files:**
- Create: `talks/booking-2026/booking-ragtune.md`

This is the 2-slide RAGtune section (6 marp slides counting steps) that slots into Act 4 of the Booking 2026 talk after the self-play loop section.

- [ ] **Step 1: Write booking-ragtune.md**

```markdown
---
marp: true
theme: default
paginate: false
size: 16:9
style: |
  :root {
    --paper: #faf7f2;
    --ink:   #1a1a1a;
    --muted: #6b665e;
    --accent: #c8553d;
    --cool:  #2a4747;
  }
  section {
    font-family: Georgia, 'Times New Roman', serif;
    background: var(--paper);
    color: var(--ink);
  }
---

<!-- _header: '' -->
<!-- _footer: '' -->

![bg fit](booking-diagrams/slide_ragtune_mapping_s1.svg)

---
<!-- _header: '' -->
<!-- _footer: '' -->

![bg fit](booking-diagrams/slide_ragtune_mapping_s2.svg)

---
<!-- _header: '' -->
<!-- _footer: '' -->

![bg fit](booking-diagrams/slide_ragtune_mapping_s3.svg)

---
<!-- _header: '' -->
<!-- _footer: '' -->

![bg fit](booking-diagrams/slide_ragtune_status_s1.svg)

---
<!-- _header: '' -->
<!-- _footer: '' -->

![bg fit](booking-diagrams/slide_ragtune_status_s2.svg)

---
<!-- _header: '' -->
<!-- _footer: '' -->

![bg fit](booking-diagrams/slide_ragtune_status_s3.svg)
```

- [ ] **Step 2: Verify file**

```bash
grep -c "^---$" /Users/avishekanand/talks/booking-2026/booking-ragtune.md
# expected: 6 (one per step slide)
grep "booking-diagrams/" /Users/avishekanand/talks/booking-2026/booking-ragtune.md | grep -v "_s[1-3]\.svg"
# expected: no output
```

- [ ] **Step 3: Commit**

```bash
cd /Users/avishekanand/talks/booking-2026
git add booking-ragtune.md
git commit -m "feat: add booking-ragtune.md RAGtune section for Booking 2026 talk"
```

---

## Self-Review Notes

**Spec coverage check:**
- ✅ A1 slide_rag_pipeline — Task 1
- ✅ A2 slide_ragtune_loop — Task 2
- ✅ A3 slide_estimator_matrix — Task 3
- ✅ A4 slide_results_pareto — Task 4 (placeholder values documented with formula)
- ✅ B1 slide_ragtune_mapping — Task 5
- ✅ B2 slide_ragtune_status — Task 6
- ✅ Marp deck integration (4 slides replaced) — Task 7
- ✅ booking-ragtune.md created — Task 8
- ✅ Directory setup + .gitignore — Task 0

**Parallel execution:** Tasks 1–4 (Part A figures) are independent and can be run in parallel. Tasks 5–6 (Part B figures) are independent and can run in parallel with each other and with Tasks 1–4. Tasks 7–8 depend on all figure tasks completing first.

**Bar chart note (Task 4):** All NDCG values are placeholders. Formula documented in the plan. Real values can be substituted without touching any other figure.
