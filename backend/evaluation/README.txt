# End-to-End Evaluation

Copy `end_to_end_dataset.json` and `evaluate_end_to_end.py` into your existing
`backend/evaluation/` folder.

If needed, create an empty `backend/evaluation/__init__.py`.

Run from the backend directory:

```powershell
python -m evaluation.evaluate_end_to_end
```

The script uses the actual `app.requirement_extractor.extract_requirements`
and `app.job_matcher.match_resume_to_job` functions. It measures requirement
extraction precision/recall/F1 and per-requirement match-status agreement.

This synthetic dataset includes 30 cases across ten roles (strong, mixed, and
unrelated resumes) plus five adversarial cases. Review all failures manually.
The dataset is intended for development/regression testing, not as a claim of
real-world accuracy.
