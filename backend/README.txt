# Requirement Extractor Evaluation

## Files
- `extractor_dataset.json`: manually labeled job-description examples.
- `evaluate_extractor.py`: runs the examples against the real extractor and reports precision, recall, F1, importance checks, and a `top_n` check.

## Install / run
Copy both files into your project's `backend/evaluation/` folder. Ensure that folder has an `__init__.py` file (create an empty one if needed), then from the `backend` directory run:

```powershell
python -m evaluation.evaluate_extractor
```

The script imports `extract_requirements` from `app.requirement_extractor`, so it evaluates your real implementation rather than a mock.

## Important
This is a small diagnostic dataset, not a statistically representative benchmark. The expected outputs are manually labeled against the current extractor's canonical aliases. If a case fails, inspect the output before changing code; some requirements may be affected by the extractor's known-term dictionary or normalization rules.
