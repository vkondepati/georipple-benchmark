.PHONY: install test bench tables all
install:
	pip install -r requirements.txt
test:
	python -m pytest -q tests
bench:
	python experiments/run_benchmark.py --seeds 5 --horizon 28
tables:
	python experiments/make_tables.py
all: test bench tables
