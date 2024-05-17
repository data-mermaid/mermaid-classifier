FROM python:3.11-bullseye
LABEL maintainer="saanobhaai"

ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8
ENV LANGUAGE C.UTF-8
ENV LC_ALL C.UTF-8
# Set environment variables to reduce Python package issues and ensure output is sent straight to the terminal without buffering it first
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
    
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    less \
    nano \
    wget \
    bzip2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 


# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Set conda environment variables
ENV PATH=/opt/miniconda/bin:$PATH
RUN conda update -n base -c defaults conda

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "mermaid-dev", "/bin/bash", "-c"]

# Demonstrate the environment is activated:
RUN conda run -n mermaid-dev python -c "import spacer"

WORKDIR /app

# The code to run when container is started:
COPY src/base.py .
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mermaid-dev", "python", "src/base.py"]