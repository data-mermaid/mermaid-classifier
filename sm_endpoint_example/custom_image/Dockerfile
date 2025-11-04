# Using as a template:
# https://github.com/niklas-palm/sagemaker-endpoint-custom-container/blob/bf8a374cf85cd7a0af72725a8cda27330ea3045b/custom_image/Dockerfile

FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install image-classification functionality
RUN pip install "mermaid-classifier[pyspacer] @ git+https://github.com/data-mermaid/mermaid-classifier.git"

# Copy the Flask application code into the container
COPY . .

# Set the environment variable for the SageMaker model directory
ENV SM_MODEL_DIR /opt/ml/model

# Expose port 8080 to allow external access to the Flask application
EXPOSE 8080

# Define the entry point for running the Flask application
ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:8080", "app:webserver"]
