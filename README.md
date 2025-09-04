# mermaid-classifier

Code for training and deploying MERMAID image classifiers.

This repo has several different kinds of runnable programs, as follows:


## Notebooks

These iPython notebooks have the most recent developments as of late 2025. These are set up to run in a SageMaker JupyterLab space. To run them:

- Sign into AWS, open SageMaker Studio, and navigate to JupyterLab spaces. Start and open the space of your choice.

- After starting up and opening the JupyterLab space, open a Terminal tab and run the following:

    `pip install pyspacer@git+https://github.com/coralnet/pyspacer.git@main "mlflow>=3" python-decouple ipympl`

    You'll see a message "ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. ...". Don't worry about it, because it should just be related to SageMaker-preinstalled packages that we don't use.

    Currently, you'll have to re-run this command once every time you shut down the space and then start it again.

- To start up the MLflow tracking server (needed primarily for training): from SageMaker Studio, navigate to MLflow, and start the tracking server designated for pyspacer classifiers.

    Note: It'll take about 20 minutes to start up, and costs about $0.60/hour (much more than the JupyterLab space) to leave up and running.

- To get interactive matplotlib working (useful for viewing/saving annotation plots): [hard-refresh](https://www.howtogeek.com/672607/how-to-hard-refresh-your-web-browser-to-bypass-your-cache/) the browser tab that has the JupyterLab space open. This must be done after pip-installing ipympl.

- In the File Browser on the left of the UI, navigate to the `mermaid-classifier` folder, then to the `notebooks` sub-folder.

- Open the notebook (.ipynb file) of your choice.

- Select "Python 3 (ipykernel)" as the notebook's kernel.

- Edit and run the notebook cell of your choice, and wait for the results.

If you just created a new JupyterLab space, you'll first have to set it up:

- Git checkout the `mermaid-classifier` repo.

- At the root of the `mermaid-classifier` repo, create a `.env` file which defines SPACER_EXTRACTORS_CACHE_DIR, MLFLOW_TRACKING_SERVER_ARN, and WEIGHTS_LOCATION.


## Make commands

These require:

- A [Docker installation](https://www.docker.com/products/docker-desktop)

- coralnet-bocas zip

### Running docker in your local machine

`docker run -v $(pwd)/cache:/workspace/cache -v $(pwd)/spacer:/workspace/spacer -it pyspacer-docker bash`

`docker build -f Dockerfile -t pyspacer .`

`docker run -it pyspacer bash`

You can run: 

`from spacer import config` 

To verify that spacer installed correctly. 

### Setup development environment

We setup the development environment in a Docker container with the following command.

`make build`

This command gets the resources for training and testing, and then prepares the Docker image for the experiments. After creating the Docker image, you run the following command.

`make run`

The above command creates a Docker container from the Docker image which we create with `make build`.

### Classify Features

To classify features, you can run the following command:

`make classify`

To create a file archive of a container image, use this command, changing the name of the archive file and container to reflect the names you want to use:

`docker save --output archive-name.tar username/imagename:tag`

It’s also a good idea to archive a copy of the Dockerfile used to generate a container image along with the file archive of the container image itself.


## Quarto documents

These `.qmd` files require:

- What the Make commands require, above

- A [Quarto installation](https://quarto.org/docs/getting-started/installation.html).

    - After installation, you can verify that Quarto is installed correctly by running:
    
      `quarto --version`

      This should display the version of Quarto installed on your machine.
