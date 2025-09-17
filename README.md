# mermaid-classifier

This Python repository enables data scientists to experiment with PySpacer-based classifiers. It also has MERMAID-relevant utilities which aren't specific to the type of classifier being developed.


## Overview

This project is set up as a Python package, and requires Python 3.10 or higher. Once you have the package installed in your Python environment, you can import anything from `mermaid_classifier` into your own Python modules, notebooks, etc.

### General utilities

These are found in `mermaid_classifier.common`. Once this package is installed (see Installation section), the utilities can be imported from there.

### PySpacer training and classification code

This is found in `mermaid_classifier.pyspacer`.

### Documentation

See the [docs](docs) section for usage explanations.

### v1 directory

This is the work from MERMAID classifier version 1 which hasn't been incorporated into the current version yet.


## Installation

### Python package installation

Some installation examples:

| Result | Command |
| - | - |
| Utilities only | `pip install https://github.com/data-mermaid/mermaid-classifier.git` |
| Utilities + PySpacer-based classification | `pip install https://github.com/data-mermaid/mermaid-classifier.git[pyspacer]` |
| Utilities + PySpacer-based classification + JupyterLab support | `pip install https://github.com/data-mermaid/mermaid-classifier.git[pyspacer,jupyterlab]` |
| Utilities only, at non-main branch | `pip install "mermaid-classifier @ git+https://github.com/data-mermaid/mermaid-classifier.git@my-branch-name"` |
| Utilities + PySpacer-based classification + JupyterLab support, at non-main branch | `pip install "mermaid-classifier[pyspacer,jupyterlab] @ git+https://github.com/data-mermaid/mermaid-classifier.git@my-branch-name"` |

To update your install, add `-U` after the word `install` in any of the above. However, if the package's version number has not been bumped up yet, you'll probably have to `pip uninstall mermaid-classifier` first, otherwise pip might think there is nothing to be updated.

If you're in a SageMaker JupyterLab space:

- After you shut down the space and then start it again, you'll have to re-run pip installations.

- Running the pip install command from a Terminal tab should work.

- At the end of the install, you'll see a message "ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. ...". That's most likely related to SageMaker-preinstalled packages that this repo doesn't deal with, so it's most likely not a concern.

### Additional steps for PySpacer classifiers

1. In the directory that you're running your code or notebook from, create a `.env` file which defines variables, like in the following example:

    ```
    SPACER_EXTRACTORS_CACHE_DIR=/path/to/extractors_cache
    MLFLOW_TRACKING_SERVER=arn:aws:sagemaker:us-east-1:123412341234:mlflow-tracking-server/my-server
    MLFLOW_DEFAULT_EXPERIMENT_NAME=Label filters and rollups
    WEIGHTS_LOCATION=/path/to/weights.pt
    ```
    
    If you're missing a required setting, you should get an error saying so when your code runs. For more information on these settings, see `mermaid_classifier/pyspacer/settings.py`.

    Alternatively, you may directly set environment variables instead of using an `.env` file.

2. If you're in JupyterLab, you need to have interactive matplotlib working to have pan, zoom, and save controls on annotation plots. If you want this, after pip-installing ipympl, you must [hard-refresh](https://www.howtogeek.com/672607/how-to-hard-refresh-your-web-browser-to-bypass-your-cache/) the browser tab that has the JupyterLab space open.

### Installation environment

Although MERMAID IC is primarily targeting an AWS SageMaker environment, this package can also be set up on a local dev machine.

AWS SageMaker advantages over local:

- Easily and securely access private S3 files through spaces, as long as the SageMaker domain is set up with an applicable Space execution role.
- Web-based IDE spaces with real-time collaboration.
- MLflow tracking servers can be shared by everyone who can access the SageMaker domain.
- Default distribution image already has many Python packages relevant to this project. This could be preferable over maintaining a 3 GB local venv.

Local env advantages over SageMaker:

- Don't have to worry about the AWS web session expiring every so often, and don't need constant internet to keep working.
- More IDE choices, not just VSCode (Code Editor spaces) or JupyterLab.
- Can run a local MLflow tracking server with very low startup and cost.
- Easier to customize and persist the packages that are installed in the environment.

If you're on a local dev machine and accessing public S3 files, the `SPACER_AWS_ANONYMOUS` setting may be useful.


## For developers

Set up this project as an [editable install](https://pip.pypa.io/en/stable/topics/local-project-installs/): first git-clone this repo, then use `pip install -e <path to repo>`.

### Design notes

This project is set up as a Python package with a [flat project layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/).

Although this project isn't on PyPI, the fact that it's set up as a package makes it easier to:

- Import from this project, compared to an ad-hoc addition to `sys.path`, for example.

- Manage project dependencies.
