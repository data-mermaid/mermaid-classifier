# mermaid-classifier

Code for training and deploying MERMAID image classifiers.


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


## v1

This directory contains code, setup, etc. that went into MERMAID's v1 classifier and hasn't been adapted to v2 yet.
