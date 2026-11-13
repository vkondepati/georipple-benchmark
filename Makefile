PYTHON ?= python3
TIGER_ZIP ?= tl_2023_us_primaryroads.zip
BUILD ?= build
FIGURES ?= $(BUILD)/figures
CONFIRM = --seeds 10 --seed-offset 10 --horizon 28

.PHONY: install test lint artifacts bench confirm sensitivity collect tables figures roads gate check all
install:
	$(PYTHON) -m pip install -r requirements.lock
test:
	$(PYTHON) -m pytest -q tests
lint:
	$(PYTHON) -m ruff check georipple experiments tests
artifacts:
	$(PYTHON) experiments/check_artifacts.py
# Raw runs go to $(BUILD)/ (ignored), so writing one never dirties another's provenance.
bench:
	$(PYTHON) experiments/run_benchmark.py --seeds 10 --horizon 28 --out $(BUILD)/runs.json
confirm:
	$(PYTHON) experiments/run_benchmark.py $(CONFIRM) --lanes straight --dealer-dcs 1 --out $(BUILD)/confirm/C0.json
	$(PYTHON) experiments/run_benchmark.py $(CONFIRM) --lanes road --dealer-dcs 1 --out $(BUILD)/confirm/C1.json
	$(PYTHON) experiments/run_benchmark.py $(CONFIRM) --lanes straight --dealer-dcs 2 --out $(BUILD)/confirm/C2.json
	$(PYTHON) experiments/run_benchmark.py $(CONFIRM) --lanes road --dealer-dcs 2 --out $(BUILD)/confirm/C3.json
sensitivity:
	$(PYTHON) experiments/run_sensitivity.py --seeds 5 --horizon 28 --out $(BUILD)/sensitivity_runs.json
collect:
	mkdir -p results/confirm
	cp $(BUILD)/runs.json $(BUILD)/sensitivity_runs.json results/
	cp $(BUILD)/confirm/C0.json $(BUILD)/confirm/C1.json $(BUILD)/confirm/C2.json $(BUILD)/confirm/C3.json results/confirm/
tables:
	$(PYTHON) experiments/make_tables.py
	for c in C0 C1 C2 C3; do $(PYTHON) experiments/make_tables.py --runs results/confirm/$$c.json --outdir results/confirm/$$c || exit 1; done
	$(PYTHON) experiments/make_conditions.py
	$(PYTHON) experiments/make_sensitivity.py
gate:
	$(PYTHON) experiments/regression_gate.py --new results/runs.json
figures:
	$(PYTHON) experiments/make_figures.py --seed 0 --hazard 0 --outdir $(FIGURES)
	$(PYTHON) experiments/make_figures.py --seed 0 --hazard 0 --lanes road --outdir $(FIGURES)
roads:
	$(PYTHON) experiments/prepare_roads.py $(TIGER_ZIP)
check: test lint artifacts
	$(PYTHON) -m compileall -q georipple experiments tests
# From a clean checkout: every raw run, then tables, the gate, and the checks.
all: bench confirm sensitivity collect tables gate figures check
