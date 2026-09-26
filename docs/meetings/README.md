# Meeting Notes

This directory stores meeting notes for the SIR / RAGtune project. The purpose is to preserve institutional knowledge, maintain alignment across team members who work asynchronously across institutions, and provide a version-controlled record of design decisions, action items, and rationale.

## File naming

Each meeting note is a single Markdown file named `YYYY-MM-DD.md` corresponding to the date of the meeting. For meetings spanning multiple sessions on the same date, append a lowercase letter suffix: `2026-05-13a.md`, `2026-05-13b.md`.

## Format

Every meeting note follows this structure:

```markdown
# YYYY-MM-DD Meeting

**Participants**: comma separated list of GitHub usernames or names
**Agenda**: brief one-line summary of the meeting purpose

## Discussion

Each agenda item is an H3 section. Discussion points use bullet points with timestamps in parentheses where relevant:

### Repository Onboarding

- Positive onboarding experience reported by Shuvam and Rahul (08:42)
  - Shuvam found xfail tests easy to learn.
  - Rahul found the repository manageable with clear instructions.

## Decisions

A bullet list of explicit decisions made during the meeting. Each decision should be linkable to the discussion section that motivated it.

- Decision: Meeting notes will be stored in `docs/meetings/` for version-controlled traceability.

## Action Items

| Action Item | Owner | Status | Related Issue/PR |
|---|---|---|---|
| Description of the action | @githubuser | Open / In Progress / Done | #issue-number or PR link |

## Related Links

- Link to GitHub Issues or PRs discussed
- Link to wiki pages referenced
- Link to external documents or shared drives
```

## Action item tracking

Action items from each meeting are listed in a table at the bottom of the note. The `Status` column tracks progress. When an action item is completed, its status is updated to `Done` and a link to the resulting artifact (PR, issue, document) is added.

For cross-meeting visibility, an aggregated view of all open action items is maintained in `ACTION_ITEMS.md` at the root of this directory.

## Exclusions from packaging

This directory is excluded from the Python wheel by the existing `[tool.setuptools.packages.find]` configuration in `pyproject.toml` which scopes package discovery to `src/`. Meeting notes are part of the source repository only; they are not installed by `pip install -e .` or `pip install ragtune`.

## Related

- [PR & Code Review Guidelines](https://github.com/avishekanand/sir/wiki/PR-and-Code-Review-Guidelines)
- [Meeting Rhythm (wiki)](https://github.com/avishekanand/sir/wiki/Meeting-Rhythm)
