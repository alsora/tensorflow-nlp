# tensorflow-text

This repository implements a Tensorflow framework for performing Text Classification and Natural Language Processing (NLP) tasks.
This repository contains scripts and libraries for different programming languages.

  - Python for creating neural networks, performing train and test, exporting network models.

  - Java for loading pretrained network models, performing inference or serving them in production.

  - Node.js for loading pretrained network models, performing test and inference.


## Python:
The Python implementation of this framework is provided with a Dockerfile, in order to be easily used on servers or other machines.
If you want to install the Framework using Docker, skip this section.

### Installation

  - Install basic requirements
  
        $ sudo apt-get install python3-pip python3-dev
  - Install Tensorflow latest version
  
        $ pip3 install tensorflow
  - Install additional requirements
  
        $ cd python/script/ && pip3 install -r requirements.txt
  - Validate installation

        $ python3


        >>> import tensorflow as tf
        >>> hello = tf.constant('Hello, TensorFlow!')
        >>> sess = tf.Session()
        >>> print(sess.run(hello))



### Docker installation

The only requirement in this case is to have [Docker](https://docs.docker.com/install/) installed  on the host machine.

The repository contains some useful scripts to simplify the creation of a Docker image and the deployment of a Docker container.

  - Export some environment variables
  
        $ source env.sh
  - Build a Docker image
  
        $ bash build.sh
  - Deploy a Docker container
  
        $ bash run.sh $VERSION


### Usage:

This framework allows to easily create Neural Networks model and train/test them on a variety of Natural Language Processing tasks.

To check if everything is working fine, it's possible to perform the train of a neural network for a sentiment classification task on the provided sample_data.

        $ cd python
        $ python train.py 

It's possible to see all the available flags using 

        $ python train.py --h



## Java:


### Installation

Follow these steps to install the Java bindings for Tensorflow.

  - Download the .jar file and the Java Native Interface (JNI).
  
        $ cd java
        $ bash script/jni.sh

  - In your preferred IDE, add the .jar file to the Java project build path and link to it the native libraries contained in the jni folder.
  Then you can validate the installation by running the HelloTF.java example.


  - How to compile and validate the Java framework from command line, if you do not want to use an IDE:

        $ javac -d bin -sourcepath src -cp lib/libtensorflow-1.8.0.jar src/main/HelloTF.java
        $ java -cp bin:lib/libtensorflow-1.8.0.jar -Djava.library.path=jni main.HelloTF


## Node.js:

### Installation

Follow these steps to install Tensorflow.js

  - Install via [npm](https://www.npmjs.com/) all the required dependencies. Note that an up-to-date version of node is required.
  
        $ cd node
        $ npm install .

  - Validate the installation by running an LSTM example.

        $ cd lib/tests/lstm
        $ node index.js

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



