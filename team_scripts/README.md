# Team scripts: one command per member

Each member runs **one** Python file. It writes that member's files (exact content, sha256-checked) and runs the tests, then prints the git commands to push.

Everyone works directly on `master`, **one after another**. No two scripts write the same file, and each member starts from the master that already has the previous push, so there are no merge conflicts.

| Order | Who | Run | What it adds |
|---|---|---|---|
| 1 | Kenan | `python team_scripts/1_kenan.py` | Tree: feature importances, branch split search (zero-gain splits), constant-target stop; `tree_viz.py`; `tests/conftest.py`; tree, ensemble, tree-viz and web tests; the website `src/web/`; report models section |
| 2 | Murad | `python team_scripts/2_murad.py` | `evaluate.py` (dead code removed), `validate.py` (correct filter text); Pegasos fixes (capped intercept step, ‖w‖ projection) + `RandomFourierPegasosSVM`; `metrics.py`; `experiments.py` (RFF SVM, 5-fold CV, learning curves, figures, web report data); `site_data.py`; SVM, metrics, validate and site-data tests; report results section |
| 3 | Emiliya | `python team_scripts/3_emiliya.py` | `explore.py` (latest listing per advert, `explore_raw`, guards); report data section |
| 4 | Aygun | `python team_scripts/4_aygun.py` | `run_all.py` gets the experiments stage; main `report.tex`, slides, `CHANGES.md`, `contribution_report.md`, `docs/team/`; then runs the full pipeline and builds the PDFs |

## Steps for every member
```bash
cd PegaSold
git checkout master
git pull origin master
python team_scripts/<n>_<name>.py
# the script ends by printing exactly what to run:
git add <your files> && git commit -m "..." && git push origin master
```
Then tell the next member to start.

## Safety checks built into each script
- **Location:** it stops unless it is run from the repo root, on `master`.
- **Clean tree:** it stops if tracked files have uncommitted changes.
- **Order:** it stops if the previous member's part is not in master yet ("wait for ... then pull"), or if it was already applied.
- **Tests:** they must pass before it prints the push commands.

Requirements: `pip install -r requirements.txt`. For the full run (step 4), `data/bina_az_sale.csv` must be present. `pdflatex` is optional: without it, the PDFs shipped inside the script are used.
