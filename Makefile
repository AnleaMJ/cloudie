.PHONY: test smoke baselines train-smoke app
test:
	pytest
smoke:
	python -m src.make_smoke_data --out data/smoke --profile balanced
baselines: smoke
	python -m src.baselines --config config/config_smoke.yaml
train-smoke:
	python -m cloudie.make_smoke_data --out data/smoke
	python -m cloudie.train --config config/config_smoke.yaml
	python -m cloudie.evaluate --config config/config_smoke.yaml
app:
	streamlit run cloudie/app.py --server.headless true
