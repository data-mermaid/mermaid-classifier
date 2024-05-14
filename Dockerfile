FROM python:3.11-bullseye
LABEL maintainer="saanobhaai"

ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8
ENV LANGUAGE C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    less \
    nano

RUN /usr/local/bin/python -m pip install --upgrade pip
RUN /usr/local/bin/python -m pip install --no-cache-dir \
    pyspacer==0.9.0

WORKDIR /app
