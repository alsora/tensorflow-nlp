## Tensorflow-NLP in Python 

The Python library is the most complete one provided in this framework.
It is intended for developers who wants to create new network models and train them, exploiting all the functionalities of Tensorflow.


This library can be fully accessed through the dedicated Docker container.

### Usage:


##### Network train

This framework supports different NLP tasks, such as text classification and sequence tagging. For each of them a specific train file is provided.

To train a bidirectional LSTM on a text classification task:

        $ python train_text_classification.py --model blstm --data ../data/dataset/sample_data/train.tsv

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
      --output_dir: Where to save the trained model, checkpoints and stats (default: pwd/runs/timestamp)
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



##### Network interactive session







