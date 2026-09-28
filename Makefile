PYTHON ?= python3

.PHONY: install test bench tables figures all
install:
	$(PYTHON) -m pip install -r requirements.txt
test:
	$(PYTHON) -m pytest -q tests
bench:
	$(PYTHON) experiments/run_benchmark.py --seeds 5 --horizon 28
tables:
	$(PYTHON) experiments/make_tables.py
figures:
	$(PYTHON) experiments/make_figures.py --seed 0 --hazard 0
all: test bench tables figures
