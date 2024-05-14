IMAGE=mermaid/mermaid-classifier


build:
	docker build --no-cache -t $(IMAGE) .

run:
	docker run --rm -it --env-file .env -v `pwd`/src:/app $(IMAGE) python task.py

shell:
	docker run -it --env-file .env -v `pwd`/src:/app $(IMAGE) bash
