.PHONY: test smoke train-smoke
test:
	pytest
smoke:
	python -m cloudie.make_smoke_data --out data/smoke
train-smoke:
	python -m cloudie.make_smoke_data --out data/smoke
	python -m cloudie.train --config config/config_smoke.yaml
	python -m cloudie.evaluate --config config/config_smoke.yaml
