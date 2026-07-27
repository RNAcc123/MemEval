# Reproducibility And Provenance

Record these values for every published experiment:

- Git revision and working-tree state.
- Dataset name, version, source URL or local acquisition identifier.
- Memory backend and configuration, including embedding model.
- Judge provider, model identifiers, temperature, retry policy, and prompt version.
- Run ID, start time, completion time, and completed/error record counts.

Secrets belong in `.env` or a deployment secret manager and must never be
written to manifests or result records. Generated traces, vector stores, logs,
and figures are reproducible artifacts; keep them outside Git unless a small
fixture or intentionally published result is needed for review.
