# tensorflow-text

This repository implements a Tensorflow framework for performing Text Classification and Natural Language Processing (NLP) tasks.


## Requirements:

A Dockerfile is provided for using the framework in the best possible way, without having to care about manually intsalling all the dependences.

If you want to exploit the Dockerfile, you only have to install [Docker](https://docs.docker.com/install/).

Otherwise you can repeat all the commands included in the Dockerfile and locally install all the requirements.
The main requirement is [Tensorflow](https://github.com/tensorflow/tensorflow).

## Installation:

The repository contains some useful scripts to simplify the creation of a Docker image and the deployment of a Docker container.

  - Export some environment variables
  
        $ source env.sh
  - Build a Docker image
  
        $ bash build.sh
  - Deploy a Docker container
  
        $ bash run.sh $VERSION


## Usage:

This framework allows to easily create Neural Networks model and train/test them on a variety of Natural Language Processing tasks.

To check if everything is working fine, it's possible to perform the train of a neural network for a sentiment classification task on the provided sample_data.

        $ cd tf-helper
        $ python train.py --model blstm


It's possible to see all the available flags using 

        $ python train.py --h



## Implemented Neural Networks:


  - Bidirectional LSTM [paper](https://link.springer.com/chapter/10.1007/978-3-319-39958-4_19).

  - Bidirectional LSTM + attention [paper](http://www.aclweb.org/anthology/P16-2034).

  - Convolutional Neural Network for text classification [paper](https://arxiv.org/pdf/1408.5882.pdf).


### TBD:

  - Characters LSTM [repo](https://github.com/charlesashby/CharLSTM).
  - C-LSTM [paper](https://arxiv.org/pdf/1511.08630.pdf).
  - Adversarial Classification [repo](https://github.com/dennybritz/models/tree/master/adversarial_text).
  - Word2vec enhanced with LSTM [repo](https://github.com/chaitjo/lstm-context-embeddings
).





