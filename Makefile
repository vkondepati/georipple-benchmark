.PHONY: install test bench tables figures all
install:
	pip install -r requirements.txt
test:
	python -m pytest -q tests
bench:
	python experiments/run_benchmark.py --seeds 5 --horizon 28
tables:
	python experiments/make_tables.py
figures:
	python experiments/make_figures.py --seed 0 --hazard 0
all: test bench tables figures
