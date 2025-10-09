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
