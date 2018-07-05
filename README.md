# tensorflow-NLP

This repository implements a Tensorflow framework for performing Text Classification and Natural Language Processing (NLP) tasks.
This repository contains scripts and libraries for different programming languages.

  - Python for creating neural networks, performing train and test, exporting network models.

  - Java for loading pretrained network models, performing inference or serving them in production.

  - Node.js for loading pretrained network models, performing test and inference.


## Python installation:
The Python implementation of this framework is provided with a Dockerfile, in order to be easily used on servers or other machines.
If you want to install the framework using Docker, skip the following steps and go to the next sub section.

  - Install requirements
  
        $ sudo apt-get install python3-pip python3-dev
        $ pip3 install tensorflow
        $ cd python/script/ && pip3 install -r requirements.txt

  - Validate installation

        $ python3


        >>> import tensorflow as tf
        >>> hello = tf.constant('Hello, TensorFlow!')
        >>> sess = tf.Session()
        >>> print(sess.run(hello))


##### Docker installation

The only requirement in this case is to have [Docker](https://docs.docker.com/install/) installed  on the host machine.

The repository contains some useful scripts to simplify the creation of a Docker image and the deployment of a Docker container.
We are going to export some environment variable, build the Docker image and then run a Docker container for this image.
  
        $ source env.sh  
        $ bash build.sh  
        $ bash run.sh $VERSION

An alternative run script is provided. This can be used instead of "run.sh" and will create a volume for all the Python directory into the Docker container. This allows to easily modify the code using your preferred editor from the host pc.

Be careful: this implies also that EVERY CHANGE TO THE CODE OR TO THE REPOSITORY THAT YOU PERFORM FROM INSIDE THE CONTAINER WILL REFLECT ON THE FILES ON THE HOST!!!

        $ bash run_dev.sh $VERSION

## Java installation:

Follow these steps to install the Java bindings for Tensorflow and exploit the provided Java library.

  - Download the .jar file and the Java Native Interface (JNI).
  
        $ cd java
        $ bash script/download_tensorflow.sh

  - In your preferred IDE, add the .jar file to the Java project build path and link to it the native libraries contained in the jni folder.
  Then you can validate the installation by running the HelloTF.java example.


  - How to compile and validate the Java framework from command line, if you do not want to use an IDE:

        $ javac -d bin -sourcepath src -cp lib/libtensorflow-1.8.0.jar src/main/HelloTF.java
        $ java -cp bin:lib/libtensorflow-1.8.0.jar -Djava.library.path=jni/$(uname -s | tr '[:upper:]' '[:lower:]') main.HelloTF


## Node.js installation:

Follow these steps to install Tensorflow.js

  - Install via [npm](https://www.npmjs.com/) all the required dependencies. Note that an up-to-date version of node is required.
  
        $ cd node
        $ npm install .

  - Validate the installation by running an LSTM example.

        $ cd lib/tests/lstm
        $ node index.js

