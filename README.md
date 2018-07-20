[![Build Status](https://travis-ci.org/alsora/tensorflow-nlp.svg?branch=master)](https://travis-ci.org/alsora/tensorflow-nlp)

# tensorflow-NLP

This repository implements a Tensorflow framework for performing Natural Language Processing (NLP) tasks.
This repository contains scripts and libraries for different programming languages.

  - **Python** for creating neural networks, performing train and test, exporting network models.

  - **Java** for loading pretrained network models, performing inference or serving them in production.

  - **Node.js** for loading pretrained network models, performing test and inference.
 

# Why tensorflow-NLP?

The aim of this framework is to allow to easily design networks for any NLP task, using modular code and always maintaining the same input/output structure.
This is extremely useful if your main use case is the deploy of a saved model in production, even if you want to access to this model using different programming languages.

The effectiveness of this framework is demonstrated by the fact it's possible to evaluate all the networks currently implemented, on different tasks, using the same script.




# Usage:

Instructions on how to install, configure and run everything are present in the README of their specific language directory.
You should start from the Python library and train any network from scratches.
Then it's possible to perform inference on it using a Python, Java o Node.js script.

Note: some naive datasets are provided in the *data/dataset/* folder. They should be used only for testing if a network is working properly as they are too small and the network will overfit or diverge.
[Here](https://github.com/alsora/tensorflow-nlp/tree/master/data) you can find some download links for real datasets.

### Some features!!!

Monitor the training procedure, thanks to logs and confusion matrices.
Save summaries for TensorBoard.

<img src="/docs/train.png" alt="Train epoch example"/>


Interactive inference in Python.

<img src="/docs/it.png" alt="Interactive inference"/>

Current state: Text Classification and Sequence Tagging tasks succesfully implemented in Python and Java.
Working on: Sequence2Sequence task, Dataset API, Node.js scripts. 


