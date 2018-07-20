# Tensorflow-NLP in Python 

The Python library is the most complete one provided in this framework.
It is intended for developers who wants to create new network models and train them, exploiting all the functionalities of Tensorflow.


This library can be fully accessed through a dedicated Docker image.

## Installation

##### Standard installation
This is NOT required if you want to use the Docker image

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


### Usage:


##### Network train

This framework supports different NLP tasks, such as text classification and sequence tagging. For each of them a specific train file is provided.

To train a bidirectional LSTM on a text classification task:

        $ python train_text_classification.py --model blstm --data ../data/dataset/sample_data/train.tsv --model_dir ../data/models/blstm

The script can be launched with several flags.  
There is a set of flags which is related to the training procedure and its common among all network instances.
Moreover it is possible to use flags also to set a network hyperparameters. Note that in this case the default value depends on the specific network model that you are using for training.

To see the list of flags:

        $ python train_text_classification.py --h

General purpose flags:


      --num_epochs: Number of training epochs
        (default: '10')
        (an integer)
      --batch_size: Batch Size
        (default: '64')
        (an integer)
      --data: Data source tab separated files. It's possible to provide more than 1 file using a comma
        (default: '../data/dataset/sample_data/train.tsv')
      --dev_sample_percentage: Percentage of the training data to use for validation
        (default: '0.1')
        (a number)
      --checkpoint_every: Save model after this many steps
        (default: '2000')
        (an integer)
      --evaluate_every: Evaluate model on dev set after this many steps
        (default: '2000')
        (an integer)
      --model_dir: Where to save the trained model, checkpoints and stats (default: pwd/runs/timestamp)
        (default: '')
      --num_checkpoints: Max number of checkpoints to store
        (default: '25')
        (an integer)
      --[no]summary: Save train summaries to folder
        (default: 'false')
      --[no]log_device_placement: Log placement of ops on devices
        (default: 'false')
      --[no]allow_soft_placement: Allow device soft device placement
        (default: 'true')


##### Network evaluation

It's possible to evaluate your just trained network using the test script.
Note that this is independent of the type of task.

In the following, assume to have a trained model into "/data/models/blstm"

In order to start an interactive session where the network will classify user keyboard input:

        $ python test.py --it 1 --model_dir ../data/models/blstm

If you want to test your model on a whole dataset, you simply have to specify its path instead of the interactive flag:

        $ python test.py --data ../data/dataset/MY_TEST_DATASET --model_dir ../data/models/blstm







