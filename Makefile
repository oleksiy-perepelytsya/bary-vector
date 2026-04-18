.PHONY: install up up-gpu down lint test test-int pipeline fixture \
        fetch-kaikki preflight clean-state

PY ?= python

install:
	$(PY) -m pip install -e ".[dev]"

up:
	docker compose up -d mongodb

up-gpu:
	docker compose --profile gpu up -d

down:
	docker compose down

lint:
	ruff check .
	mypy

test:
	pytest --cov=lib --cov-report=term

test-int:
	pytest -m integration -v

STAGES = s01_parse s02_embed s03_insert_nodes s04_l15_edges s05_word_vectors \
         s06_l14_edges s07_orphan_reentry s08_metabary s10_index

pipeline:
	@for s in $(STAGES); do \
	  echo "== $$s =="; \
	  $(PY) -m scripts.$$s || exit 1; \
	done

fixture:
	$(PY) -m scripts.dev.make_fixture

fetch-kaikki:
	bash scripts/dev/fetch_kaikki.sh

preflight:
	$(PY) -m scripts.preflight

clean-state:
	rm -f pipeline_state/*.json
	rm -f data/parsed/*.jsonl
