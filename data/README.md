# Data And Runs

`data/` is split by lifecycle:

- `data/samples/`: small reviewable fixtures used by tests and examples.
- `data/input/`: source datasets and generated traces consumed by diagnosis.
- `data/output/`: diagnosis results, metrics, reports, and figures.
- `runs/`: append-only run directories created by `JsonlRunStore`.

Large source datasets and model outputs are not required for the offline test
suite. Keep them outside Git or publish them through an artifact registry.
Never commit API keys, local vector stores, temporary locks, or machine-specific
absolute paths.

Every published trace should record its dataset, memory backend, model,
configuration, source revision, and generation timestamp. New runs may use:

```text
runs/<run_id>/manifest.json
runs/<run_id>/results.jsonl
runs/<run_id>/errors.jsonl
runs/<run_id>/summary.json
```

Historical JSON arrays remain read-only inputs and are never rewritten during
migration.
