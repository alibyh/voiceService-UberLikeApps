.PHONY: install index dev eval test clean

install:
	pip install -r requirements.txt

index:
	python -m matcher.index_build

dev:
	uvicorn matcher.api:app --reload --host 0.0.0.0 --port 8000

eval:
	python -m matcher.eval

test:
	pytest -q

clean:
	rm -rf matcher/data/index/*.pkl matcher/data/index/*.idx matcher/data/index/*.npy matcher/data/index/metadata.json
