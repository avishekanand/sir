---
name: ship
description: Pre-push quality gate. Runs tests, audits dead code and stale tests, writes a detailed commit message, and pushes to remote.
disable-model-invocation: true
allowed-tools: Bash, Read, Grep, Glob
---

You are running the /ship quality gate for RAGtune. Work through every step below in order. Stop at Step 1 if tests fail.

## Step 1 — Run the full test suite

Run `pytest tests/unit -q`.

If `tests/integration/` has test files, run `pytest tests/integration -q` too.

If ANY test fails: print the failure, explain the likely cause, and STOP. Do not commit or push.

If all pass: report the count and continue.

## Step 2 — Audit for dead code and stale artifacts

Check each category. For each issue print: `[CATEGORY] file:line — description`
If clean, print: `✓ <category> — none found`

**2a. Dead imports in src/**
Check `from ragtune...` imports in `src/`, `scripts/`, and `examples/`. Flag any that reference a module path that no longer exists under `src/`. (e.g. `ragtune.utils.console` was deleted — flag if still imported anywhere.)

**2b. Unused @registry components**
List any class decorated with `@registry.*` in `src/ragtune/components/` that is never imported in `tests/`, `examples/`, `scripts/`, or `src/ragtune/cli/`.

**2c. Outdated examples and scripts**
For each `.py` file in `examples/` and `scripts/`, verify every `from ragtune` import resolves to a file that still exists under `src/`. Flag any that don't.

**2d. Orphaned test files**
For each file in `tests/unit/` and `tests/integration/`, infer the module under test from the filename (e.g. `test_rerankers.py` → `src/ragtune/components/rerankers.py`). Flag test files whose source no longer exists. Also flag any import inside a test file that references a class or function that no longer exists in the source.

**2e. TODO / FIXME inventory**
Run `grep -rn "TODO\|FIXME\|HACK\|XXX" src/ tests/ examples/ scripts/ 2>/dev/null` and list every hit. These are informational, not blockers.

## Step 3 — Stage and write commit

1. Run `git status` to see what is modified and untracked.

2. Stage all modified tracked files with `git add -u`. Also stage any clearly in-scope untracked files (new source files, new tests, new examples, new skill/command files).

   Do NOT stage: `results/`, `data/`, `*.csv`, `*.json` experiment outputs, `.env`, or `ReformIR-main/`.

3. Run `git diff --cached` to confirm exactly what will be committed.

4. Write a commit message using this structure:

   ```
   <type>(<scope>): <imperative summary under 72 chars>

   ## What changed
   - <one bullet per logical change, be specific>

   ## Why
   <1-3 sentences on the problem solved or feature added>

   ## Audit
   - Tests: <N passed, integration: N passed / skipped>
   - Dead imports: <found: list | none>
   - Unused components: <found: list | none>
   - Stale tests: <found: list | none>
   - TODOs logged: <N>

   Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
   ```

   Valid types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`.

5. Create the commit with a HEREDOC so formatting is preserved:
   ```bash
   git commit -m "$(cat <<'EOF'
   <message here>
   EOF
   )"
   ```

## Step 4 — Push

Run `git push`. Report the remote and branch pushed to.

## Step 5 — Final summary table

Print this table filled in:

| Check              | Result                        |
|--------------------|-------------------------------|
| Unit tests         | N passed                      |
| Integration tests  | N passed / skipped            |
| Dead imports       | found (list) / none           |
| Unused components  | found (list) / none           |
| Stale tests        | found (list) / none           |
| Outdated examples  | found (list) / none           |
| TODOs logged       | N                             |
| Committed          | `<hash> <subject>`            |
| Pushed             | `origin/main`                 |
