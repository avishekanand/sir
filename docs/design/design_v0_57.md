# RAGtune Core 0.57 Design Specification: Pipeline Visualization & Interactive Editing

This document describes the pipeline visualization and interactive editing capabilities introduced in v0.57. These features enable users to visually understand their pipeline configuration and make changes through an interactive CLI interface.

## 1. Goal

The primary objective of v0.57 is to provide **visual comprehension and rapid iteration** on pipeline configurations. Users can:
1. Visualize the data flow through pipeline components via ASCII diagrams
2. Interactively edit component types and parameters
3. Preview changes via unified diff before saving
4. Maintain configuration sovereignty by modifying the YAML directly

## 2. The `visualize` Command

### A. Basic Usage (Read-Only)

```bash
ragtune visualize ragtune_config.yaml
```

Displays an ASCII flow diagram showing:
- Main pipeline flow: Retriever → Reformulator → Reranker → Assembler
- Support components: Scheduler and Estimator (feedback loop)
- Current budget limits

### B. Interactive Editing Mode

```bash
ragtune visualize ragtune_config.yaml --edit
```

Enables interactive editing with:
- Component-level type selection
- Parameter add/modify/remove operations
- Diff preview before save
- Optional alternate output path (`--output`)

## 3. ASCII Flow Diagram Design

The visualization renders a box-and-arrow diagram representing the pipeline:

```
╭─────────────────── RAGtune Pipeline: My Pipeline ────────────────────╮
│                                                                      │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐│
│  │ RETRIEVER  │───▶│REFORMULATOR│───▶│  RERANKER  │───▶│ ASSEMBLER  ││
│  ├────────────┤    ├────────────┤    ├────────────┤    ├────────────┤│
│  │ type: bm25 │    │ type: llm  │    │ type: cross│    │ type: greed││
│  └────────────┘    └────────────┘    └────────────┘    └────────────┘│
│                                              ▲                       │
│                    ┌────────────┐    ┌───────┴────┐                  │
│                    │ SCHEDULER  │◀───│  ESTIMATOR │                  │
│                    ├────────────┤    ├────────────┤                  │
│                    │ graceful   │    │ type: base │                  │
│                    └────────────┘    └────────────┘                  │
│                                                                      │
│  Budget: tokens=5000 | rerank_docs=50 | latency_ms=2000              │
╰──────────────────────────────────────────────────────────────────────╯
```

### Component Box Structure

Each component is rendered as a fixed-width box showing:
- Component name (uppercase)
- Current type (truncated if necessary)

## 4. Interactive Editor Flow

### A. Main Menu

```
Edit Pipeline
─────────────
1. Edit Retriever
2. Edit Reformulator
3. Edit Reranker
4. Edit Assembler
5. Edit Scheduler
6. Edit Estimator
7. Edit Budget
─────────────
d. Show diff
s. Save and exit
q. Quit without saving

Select option: _
```

### B. Component Edit Flow

1. Display current type and parameters
2. List available types from registry
3. Prompt: Change type? → select from registry choices
4. Prompt: Edit params? → add/modify/remove loop
5. Return to main menu (re-render visualization)

### C. Parameter Editing

```
Current Parameters:
  model_name: deepseek-r1:8b
  batch_size: 10

a=add, m=modify, r=remove, d=done
Action: _
```

### D. Diff Preview

Before saving, a colored unified diff is displayed:
- Green (`+`): Additions
- Red (`-`): Removals
- Cyan (`@@`): Context markers

## 5. Implementation

### A. Module Structure

```
src/ragtune/cli/
├── main.py          # visualize command entry point
└── visualize.py     # Core visualization & editing logic (~300 lines)
```

### B. Key Components

| Class/Function | Purpose |
|---------------|---------|
| `ComponentBox` | Renders single component as ASCII box |
| `PipelineFlowRenderer` | Renders complete pipeline visualization |
| `render_pipeline_flow()` | Display the visualization |
| `run_interactive_editor()` | Main editing loop |
| `edit_component()` | Component type/param editing |
| `edit_params()` | Parameter add/modify/remove |
| `edit_budget()` | Budget limits editing |
| `show_diff()` | Colored unified diff |
| `parse_value()` | Auto-detect int/float/bool/string |
| `get_available_types()` | Query registry for valid types |

### C. Dependencies

All dependencies are already present in the RAGtune stack:
- `rich`: Panel, Prompt, Confirm for interactive UI
- `pyyaml`: Configuration serialization
- `difflib` (stdlib): Unified diff generation

## 6. Command Options

```bash
ragtune visualize <config_path> [OPTIONS]

Arguments:
  config_path  Path to the pipeline configuration file

Options:
  -e, --edit          Enable interactive editing mode
  -o, --output PATH   Output path for modified config (default: overwrite input)
```

## 7. Invariants

1. **Non-Destructive by Default**: Without `--edit`, the command is read-only
2. **Diff Before Save**: Changes are always previewed before writing
3. **Registry Awareness**: Available component types are sourced from the live registry
4. **Type Coercion**: Parameter values are auto-detected (int, float, bool, string)
5. **Configuration Sovereignty**: All edits modify the YAML file directly; no external state

## 8. Testing

Unit tests cover:
- `ComponentBox` rendering correctness
- `PipelineFlowRenderer` output structure
- `parse_value()` type detection
- `get_available_types()` registry lookup
- `show_diff()` output formatting
- Edit functions with mocked prompts

```bash
pytest tests/unit/cli/test_visualize.py -v
```
