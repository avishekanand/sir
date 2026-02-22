---
description: >
  Full pre-push quality gate: run tests, audit for dead code / stale
  examples / orphaned tests, write a detailed commit, then push.
disable-model-invocation: true
allowed-tools: Bash, Read, Grep, Glob
---

You are running the /ship quality gate for RAGtune. Work through every step
below in order. Do not skip steps. If Step 1 fails, stop immediately.

## Current state

Git status: !`git status --short`

Staged diff (truncated): !`git diff --cached | head -300`

Unstaged diff (truncated): !`git diff | head -300`

Recent commits (style reference): !`git log --oneline -8`

## Step 1 — Run the full test suite

Run `pytest tests/unit -q` first.

If `tests/integration/` contains any test files, run those too with
`pytest tests/integration -q`.

- If ANY test fails: print the failure, explain the likely cause, and
  **stop here**. Do not commit or push.
- If all pass: report the count (e.g. "99 passed") and continue.

## Step 2 — Audit for dead code and stale artifacts

Check each category. For each issue print:
  `[CATEGORY] file:line — description`
If a category is clean, print: `✓ <category> — none found`

### 2a. Dead imports in src/
For each `from ragtune...` or `import ragtune...` in `src/`, verify the
referenced module path still exists as a file under `src/`. Flag any that
point to deleted or renamed files.

Also check `scripts/` and `examples/` for imports of deleted modules
(e.g., `ragtune.utils.console` was deleted — flag if still referenced).

### 2b. Unused @registry components
List any class decorated with `@registry.*` in `src/ragtune/components/`
that is never imported in `tests/`, `examples/`, `scripts/`, or
`src/ragtune/cli/`. These may be candidates for deletion.

### 2c. Outdated examples and scripts
For each `.py` file in `examples/` and `scripts/`:
- Verify every `from ragtune` import resolves under `src/`.
- Flag any hardcoded model names or class names in comments that no longer
  match the current code (e.g., a comment referencing a deleted class).

### 2d. Orphaned test files
For each file in `tests/unit/` and `tests/integration/`, infer the module
being tested from the file name (e.g., `test_rerankers.py` →
`src/ragtune/components/rerankers.py`). Flag any test file whose
corresponding source file no longer exists.

Also check each `from ragtune...` import inside test files — flag any that
reference a class or function that no longer exists in the source.

### 2e. TODO / FIXME inventory
Run: `grep -rn "TODO\|FIXME\|HACK\|XXX" src/ tests/ examples/ scripts/ 2>/dev/null`
List every hit. These are not blockers but must be visible at ship time.

## Step 3 — Stage and write commit

1. Stage all modified tracked files:
   `git add -u`

   Also stage any clearly in-scope untracked files (new source, new tests,
   new examples, new skill files). **Do NOT stage**:
   - `results/`, `data/`, `*.csv`, `*.json` (experiment outputs)
   - `.env` or any secrets file
   - `ReformIR-main/` (external repo copy)

2. Re-read `git diff --cached` to confirm exactly what is staged.

3. Write a commit message in this structure:

   ```
   <type>(<scope>): <imperative summary ≤72 chars>

   ## What changed
   - <one bullet per logical change, be specific>

   ## Why
   <1–3 sentences: what problem this solves or feature this adds>

   ## Audit
   - Tests: <N passed, integration: N passed / skipped>
   - Dead imports: <found: list them | none>
   - Unused components: <found: list them | none>
   - Stale tests: <found: list them | none>
   - TODOs logged: <N>

   Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
   ```

   Valid types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`.

4. Create the commit using a HEREDOC:
   ```bash
   git commit -m "$(cat <<'EOF'
   <your message here>
   EOF
   )"
   ```

## Step 4 — Push

Run `git push`. Report the remote and branch.

## Step 5 — Final summary table

| Check                | Result                              |
|----------------------|-------------------------------------|
| Unit tests           | N passed                            |
| Integration tests    | N passed / skipped                  |
| Dead imports         | found (list) / none                 |
| Unused components    | found (list) / none                 |
| Stale tests          | found (list) / none                 |
| Outdated examples    | found (list) / none                 |
| TODOs logged         | N                                   |
| Committed            | `<hash> <subject>`                  |
| Pushed               | `origin/main`                       |
