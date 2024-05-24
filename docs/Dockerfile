FROM python:3.11-bullseye
LABEL maintainer="yeelauren"

# Set environment variables to reduce Python package issues and ensure output is sent straight to the terminal without buffering it first
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LANGUAGE=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT_NAME=mermaid-viz

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    less \
    nano \
    curl \
    gdebi \
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

# Install Quarto
ARG QUARTO_VERSION="1.4.554"
RUN curl -o quarto-linux-amd64.deb -L https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-amd64.deb && \
    gdebi --non-interactive quarto-linux-amd64.deb && \
    rm quarto-linux-amd64.deb

RUN quarto install tinytex
# Create the environment
COPY environment-doc.yml .
RUN conda env create -f environment-doc.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "mermaid-viz", "/bin/bash", "-c"]

# Verify the environment is activated
RUN echo ${ENVIRONMENT_NAME}
RUN conda run -n mermaid-viz python -c "import spacer"

WORKDIR /app

# Copy the application code
COPY . .

# Set the entrypoint
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mermaid-viz", "bash"]
