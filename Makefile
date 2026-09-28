PYTHON ?= python3
TECTONIC ?= tectonic

.PHONY: install test lint artifacts bench tables figures paper check all
install:
	$(PYTHON) -m pip install -r requirements.lock
test:
	$(PYTHON) -m pytest -q tests
lint:
	$(PYTHON) -m ruff check georipple experiments tests
artifacts:
	$(PYTHON) experiments/check_artifacts.py
bench:
	$(PYTHON) experiments/run_benchmark.py --seeds 5 --horizon 28
tables:
	$(PYTHON) experiments/make_tables.py
figures:
	$(PYTHON) experiments/make_figures.py --seed 0 --hazard 0
paper:
	cd paper && $(TECTONIC) georipple_paper.tex
check: test lint artifacts
	$(PYTHON) -m compileall -q georipple experiments tests
all: check bench tables figures
