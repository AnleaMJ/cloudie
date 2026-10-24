.PHONY: test smoke
test:
	pytest
smoke:
	python -m cloudie.make_smoke_data --out data/smoke
