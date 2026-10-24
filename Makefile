.PHONY: test smoke baselines

test:
	pytest

smoke:
	python -m src.make_smoke_data --out data/smoke --profile balanced

baselines: smoke
	python -m src.baselines --config config/config_smoke.yaml
