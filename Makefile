.PHONY: test train audit lint typecheck

test:
	pytest tests/ -v --cov=certiqnet --cov-report=term-missing -n auto

train:
	python certiqnet/scripts/train.py trainer=fast_dev logger=csv model=certiq_index

audit:
	python run.py --study main_queueing --dry-run

lint:
	ruff check certiqnet/ tests/

typecheck:
	mypy --strict certiqnet/
