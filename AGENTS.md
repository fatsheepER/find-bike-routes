## Agent skills

### Issue tracker

Issues are tracked as local Markdown files under `.scratch/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the five canonical triage labels. See `docs/agents/triage-labels.md`.

### Domain docs

This repo uses a single-context layout. See `docs/agents/domain.md`.

### Verification

For each implementation ticket, run the test files named by the ticket and any directly affected test files. Treat that scoped run as the ticket's completion gate.

Run the full pytest suite once at the feature boundary, after all implementation tickets in that feature are complete, or when the user explicitly requests it. This overrides workflow skills that normally run the full suite after every ticket.

For `ready-for-human` tickets, run only the validation named by the ticket.
