---
name: ship
description: >
  Full pre-commit quality gate for RAGtune: runs tests, audits for dead code /
  outdated examples / stale tests, writes a detailed commit message, then pushes.
  Use before any merge to main.
disable-model-invocation: true
allowed-tools: Bash, Read, Grep, Glob
---

# /ship — RAGtune Quality Gate & Push

Run this before pushing any batch of work. It gates on tests, audits the
codebase for rot, writes a precise commit, and pushes.

## Context available to you right now

```
git status:
!`git status --short`

staged diff (first 200 lines):
!`git diff --cached | head -200`

unstaged diff (first 200 lines):
!`git diff | head -200`

recent commits (for style reference):
!`git log --oneline -8`

test files present:
!`find tests/ -name "*.py" | sort`

example/script files present:
!`find examples/ scripts/ -name "*.py" | sort 2>/dev/null`
```

---

## Step 1 — Run the full test suite

Run `pytest tests/unit -q` and `pytest tests/integration -q` (skip
integration if the directory is empty or absent).

- If any tests fail, **stop immediately**. Report which tests failed and
  what the likely cause is. Do NOT proceed to commit.
- If all pass, report the count and move on.

---

## Step 2 — Audit for dead code and stale artifacts

Check each of the following. For each issue found, print a bullet with the
file, line number, and a one-line description. If nothing found in a
category, say "✓ none found".

### 2a. Dead imports
Search `src/` for `import` statements that reference modules or names that
no longer exist in the project (e.g., deleted files, renamed classes).
Use `grep` + `glob` to cross-reference.

### 2b. Unused registry decorators
Search `src/ragtune/components/` for `@registry.*` classes that are never
imported in any example, test, CLI, or adapter file.

### 2c. Outdated examples
For each file in `examples/` and `scripts/`:
- Check that every `from ragtune...` import resolves to a file that still
  exists under `src/`.
- Check that any hardcoded model names or dataset IDs mentioned in comments
  still match what the code actually uses.

### 2d. Stale or orphaned tests
For each file in `tests/`:
- Check that the module it is testing (inferred from the file name, e.g.
  `test_estimator.py` → `estimators.py`) still exists under `src/`.
- Check that every class/function name imported in the test file still
  exists in the source.

### 2e. TODO / FIXME comments
`grep -rn "TODO\|FIXME\|HACK\|XXX" src/ tests/ examples/ scripts/` —
list them so they are visible at ship time.

---

## Step 3 — Stage and write commit message

1. Stage all modified and new tracked files:
   ```
   git add -u
   ```
   Also stage any untracked files that are clearly part of the current work
   (new source files, new tests, new examples). **Do not stage**:
   - `*.csv`, `*.json`, `data/`, `results/` — experiment outputs
   - `.env`, secrets
   - `ReformIR-main/` — external repo copy

2. Re-read `git diff --cached` to see exactly what will be committed.

3. Write a commit message following this structure:
   ```
   <type>(<scope>): <short imperative summary under 72 chars>

   ## What changed
   - <bullet for each logical change>

   ## Why
   <1-3 sentences on motivation — the problem solved or feature added>

   ## Audit
   - Tests: <N passed, 0 failed>
   - Dead code: <summary of findings or "none">
   - Stale tests: <summary or "none">

   Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
   ```

   Valid `<type>` values: `feat`, `fix`, `refactor`, `test`, `docs`,
   `chore`, `perf`.

4. Create the commit using a HEREDOC so formatting is preserved exactly.

---

## Step 4 — Push

Run `git push`. Report the remote URL and branch that was pushed to.

---

## Step 5 — Final summary

Print a short summary table:

| Check | Result |
|-------|--------|
| Unit tests | N passed |
| Integration tests | N passed / skipped |
| Dead imports | found / none |
| Stale tests | found / none |
| Outdated examples | found / none |
| TODOs logged | N |
| Committed | `<short hash> <subject>` |
| Pushed | `origin/main` |
