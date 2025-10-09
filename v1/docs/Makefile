IMAGE ?= mermaid/docs

build:
	docker build --no-cache -t $(IMAGE) .

run:
	docker run --rm -it --env-file ../.env -v `pwd`/docs:/app $(IMAGE) 