PYTHON ?= python3
TECTONIC ?= tectonic
TIGER_ZIP ?= tl_2023_us_primaryroads.zip
CONFIRM = --seeds 10 --seed-offset 10 --horizon 28

.PHONY: install test lint artifacts bench confirm sensitivity tables figures roads paper check all
install:
	$(PYTHON) -m pip install -r requirements.lock
test:
	$(PYTHON) -m pytest -q tests
lint:
	$(PYTHON) -m ruff check georipple experiments tests
artifacts:
	$(PYTHON) experiments/check_artifacts.py
bench:
	$(PYTHON) experiments/run_benchmark.py --seeds 10 --horizon 28
confirm:
	$(PYTHON) experiments/run_benchmark.py $(CONFIRM) --lanes straight --dealer-dcs 1 --out results/confirm/C0.json
	$(PYTHON) experiments/run_benchmark.py $(CONFIRM) --lanes road --dealer-dcs 1 --out results/confirm/C1.json
	$(PYTHON) experiments/run_benchmark.py $(CONFIRM) --lanes straight --dealer-dcs 2 --out results/confirm/C2.json
	$(PYTHON) experiments/run_benchmark.py $(CONFIRM) --lanes road --dealer-dcs 2 --out results/confirm/C3.json
sensitivity:
	$(PYTHON) experiments/run_sensitivity.py --seeds 5 --horizon 28
	$(PYTHON) experiments/make_sensitivity.py
tables:
	$(PYTHON) experiments/make_tables.py
	for c in C0 C1 C2 C3; do $(PYTHON) experiments/make_tables.py --runs results/confirm/$$c.json --outdir results/confirm/$$c; done
	$(PYTHON) experiments/make_conditions.py
figures:
	$(PYTHON) experiments/make_figures.py --seed 0 --hazard 0
	$(PYTHON) experiments/make_figures.py --seed 0 --hazard 0 --lanes road
roads:
	$(PYTHON) experiments/prepare_roads.py $(TIGER_ZIP)
paper:
	cd paper && $(TECTONIC) georipple_paper.tex
check: test lint artifacts
	$(PYTHON) -m compileall -q georipple experiments tests
all: check bench confirm tables sensitivity figures
