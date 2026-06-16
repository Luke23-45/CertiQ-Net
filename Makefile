.PHONY: test train audit lint typecheck

test:
	pytest tests/ -v --cov=certiqnet --cov-report=term-missing -n auto

train:
	python certiqnet/scripts/train.py trainer=fast_dev logger=csv model=certiq_index env=family_a

audit:
	python certiqnet/scripts/audit_state_bank.py

lint:
	ruff check certiqnet/ tests/

typecheck:
	mypy --strict certiqnet/
