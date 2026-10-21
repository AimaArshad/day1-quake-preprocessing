# Day 2-3 Lab: Part 2 additions

Copy these files into your Day 1 project (same folder structure). Then update your environment once:

```bash
conda env update -f environment.yml --prune
```

| File | Task |
|---|---|
| `quakes/etl.py` | A: ETL with logging |
| `quakes/models.py` | B: estimator and model factories (also edit `transform.build_preprocessor`) |
| `quakes/evaluate.py` | C: metrics |
| `quakes/train.py` | D: training command line |
| `notebooks/day2-3_part1_sklearn_basics.ipynb` | Part 1 (do this first) |

```bash
pytest tests/test_etl.py        # Task A
pytest tests/test_models.py     # Task B
pytest tests/test_evaluate.py   # Task C
python -m quakes.etl --mode batch
python -m quakes.train --task regression --model linear
python -m quakes.train --task classification --model logistic
```

Create `ANSWERS_ML.md` with your answers to the six lab questions in the handout.

## Scheduling note (write in this file)
Add the exact cron line and the Windows Task Scheduler steps that would run
`python -m quakes.etl --mode batch` every hour.
