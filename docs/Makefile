IMAGE ?= mermaid/docs

build:
	docker build --no-cache -t $(IMAGE) .

run:
	docker run --rm -it --env-file ../.env -v `pwd`:/app $(IMAGE) 

pre_process_labels:
	docker run --rm -it --env-file ../.env -v `pwd`:/app $(IMAGE) quarto render training_preprocessing_new_labels.qmd