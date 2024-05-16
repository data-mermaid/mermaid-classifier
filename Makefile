IMAGE ?= mermaid/mermaid-classifier
FEATURE_VECTOR_BUCKET ?= "coralnet-mermaid-share"
PARQUET_FILE ?= "pyspacer-test/allsource/selected_subsample_ModRes_reduced.parquet"

MERMAID_S3_BUCKET ?= "pyspacer-test"

build:
	docker build --no-cache -t $(IMAGE) .

run:
	docker run --rm -it --env-file .env -v `pwd`/src:/app $(IMAGE) python task.py

shell:
	docker run -it --env-file .env -v `pwd`/src:/app $(IMAGE) bash
