# tensorflow-text

This repository implements a Tensorflow framework for performing Text Classification and Natural Language Processing (NLP) tasks.


## Requirements:

A Dockerfile is provided for using the framework in the best possible way, without having to care about manually intsalling all the dependences.

If you want to exploit the Dockerfile, you only have to install [Docker](https://docs.docker.com/install/).

Otherwise you can repeat all the commands included in the Dockerfile and locally install all the requirements.

## Installation:

The repository contains some useful scripts to simplify the creation of a Docker image and the deployment of a Docker container.

  - Export some environment variables
  
        $ source env.sh
  - Build a Docker image
  
        $ bash build.sh
  - Deploy a Docker container
  
        $ bash run.sh $VERSION
