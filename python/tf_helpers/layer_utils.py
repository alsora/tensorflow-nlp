from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import FastText
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

def get_dataset(input_x, input_y, FLAGS):
    dataset = tf.data.Dataset.from_tensor_slices((input_x, input_y))
    dataset  = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.repeat()

    return dataset


def add_word_embedding_layer(model, reversed_dict):

    # Check required hyperparams
    try:
        glove_embedding_path = model.hyperparams["glove_embedding"]
    except KeyError:
        glove_embedding_path = ''

    try:
        fasttext_embedding_path = model.hyperparams["fasttext_embedding"]
    except KeyError:
        fasttext_embedding_path = ''

    try:
        embedding_dim = model.hyperparams["embedding_dim"]
    except KeyError:
        embedding_dim = 200


    if glove_embedding_path:
        init_embeddings = tf.constant(get_glove_embedding(reversed_dict, glove_embedding_path), dtype=tf.float32)
        model.embeddings = tf.get_variable("embeddings", initializer=init_embeddings, trainable=False)
    elif fasttext_embedding_path:
        init_embeddings = tf.constant(get_fasttext_embedding(reversed_dict, fasttext_embedding_path), dtype=tf.float32)
        model.embeddings = tf.get_variable("embeddings", initializer=init_embeddings, trainable=False)
    else:
        vocab_size = len(reversed_dict)
        init_embeddings = tf.random_uniform([vocab_size, embedding_dim])
        model.embeddings = tf.get_variable("embeddings", initializer=init_embeddings, trainable=True)

    model.data_embedding = tf.nn.embedding_lookup(model.embeddings, model.x)

    return model.data_embedding


def add_1d_conv_layer(model, input):

    # Check required hyperparams
    try:
        filter_sizes = model.hyperparams["filter_sizes"]
    except KeyError:
        filter_sizes = [3,4,5]

    try:
        num_filters = model.hyperparams["num_filters"]
    except KeyError:
        num_filters = 128

    input_tokens_dim = input.get_shape().as_list()[-1]
    input = tf.expand_dims(input, -1)
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        # Convolution Layer
        filter_shape = [filter_size, input_tokens_dim, 1, num_filters]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        conv = tf.nn.conv2d(
            input,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        # Maxpooling over the outputs
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, model.sequence_length_int - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
        pooled_outputs.append(pooled)

    # Combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    model.h_pool = tf.concat(pooled_outputs, 3)
    model.h_pool_flat = tf.reshape(model.h_pool, [-1, num_filters_total])

    return model.h_pool_flat




def add_birnn_layer(model, input):

    # Check required hyperparams
    try:
        num_cells = model.hyperparams["num_cells"]
    except KeyError:
        num_cells = 50

    try:
        num_layers = model.hyperparams["num_layers"]
    except KeyError:
        num_layers = 1


    fw_cells = [rnn.BasicLSTMCell(num_cells) for _ in range(num_layers)]
    bw_cells = [rnn.BasicLSTMCell(num_cells) for _ in range(num_layers)]
    fw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=model.dropout_keep_prob) for cell in fw_cells]
    bw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=model.dropout_keep_prob) for cell in bw_cells]

    model.rnn_outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(
        fw_cells, bw_cells, input, sequence_length=model.x_len, dtype=tf.float32)
    
    return model.rnn_outputs


def add_fully_connected_layer(model, input):
    
    #with tf.variable_scope(scope_name):
    W = tf.get_variable("W", shape=[input.shape[1], model.num_classes], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("b", shape=[model.num_classes], dtype=tf.float32, initializer=tf.zeros_initializer())
    
    model.l2_loss += tf.nn.l2_loss(W)
    model.l2_loss += tf.nn.l2_loss(b)

    model.logits = tf.nn.xw_plus_b(input, W, b, name="logits")

    return model.logits

def add_attention_layer(model, input):
    
    model.attention_score = tf.nn.softmax(tf.contrib.slim.fully_connected(input, 1))
    model.attention_out = tf.squeeze(
        tf.matmul(tf.transpose(input, perm=[0, 2, 1]), model.attention_score), axis=-1)
    
    return model.attention_out

def compute_predictions(model, input):

    #with tf.variable_scope(scope_name):
        
    model.predictions = tf.argmax(model.logits, -1, output_type=tf.int32, name="predictions")

    return model.predictions


def compute_softmax_loss(model, logits, labels, l2_loss = 0.0):

    # Check required hyperparams
    try:
        l2_reg_lambda = model.hyperparams["l2_reg_lambda"]
    except KeyError:
        l2_reg_lambda = 0
    
    losses =  tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
    model.loss = tf.add(tf.reduce_mean(losses), l2_reg_lambda * l2_loss, name="loss")

    return model.loss


def apply_backpropagation(model, loss, optimizer_name = 'adam', learning_rate = 1e-3):

    optimizer_name = optimizer_name.lower()

    if optimizer_name == 'adam':
        opt = tf.train.AdamOptimizer(learning_rate)
    elif optimizer_name == 'adagrad':
        opt = tf.train.AdagradOptimizer(learning_rate)
    elif optimizer_name == 'sgd':
        opt = tf.train.GradientDescentOptimizer(learning_rate)
    elif optimizer_name == 'rmsprop':
        opt = tf.train.RMSPropOptimizer(learning_rate)
    else:
        raise NotImplementedError("Unknown optimizer name: {}".format(optimizer_name))
    
    model.grads_and_vars = opt.compute_gradients(loss)
    model.optimizer = opt.apply_gradients(model.grads_and_vars, global_step=model.global_step, name="optimizer")

def compute_accuracy(model, predictions, labels):
    
    correct_predictions = tf.equal(predictions, labels)
    model.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


def compute_confusion_matrix(model, predictions, labels, num_classes):

    # Compute a per-batch confusion matrix
    batch_confusion = tf.confusion_matrix(labels=labels, predictions=predictions, num_classes=num_classes)
    # Create an accumulator variable to hold the counts
    model.confusion = tf.get_variable('confusion', shape=[num_classes, num_classes], dtype=tf.int32, initializer=tf.zeros_initializer())
    # Create the update op for doing a "+=" accumulation on the batch
    model.confusion_update = tf.assign( model.confusion, model.confusion + batch_confusion, name='confusion_update')




def get_glove_embedding(reversed_dict, glove_file):
    print("Loading Glove vectors...")
    word2vec_file = get_tmpfile("word2vec_format.vec")
    glove2word2vec(glove_file, word2vec_file)
    word_vectors = KeyedVectors.load_word2vec_format(word2vec_file)

    embedding_size = word_vectors.vector_size

    word_vec_list = list()
    for _, word in sorted(reversed_dict.items()):
        try:
            word_vec = word_vectors.word_vec(word)
        except KeyError:
            word_vec = np.zeros([embedding_size], dtype=np.float32)

        word_vec_list.append(word_vec)

    print ("Done!")
    return np.array(word_vec_list)


def get_fasttext_embedding(reversed_dict, fasttext_file):
    print("Loading Fasttext vectors...")
    model = FastText.load_fasttext_format(fasttext_file)

    embedding_size = model.vector_size

    word_vec_list = list()
    for _, word in sorted(reversed_dict.items()):
        try:
            word_vec = model[word]
        except KeyError:
            word_vec = np.zeros([embedding_size], dtype=np.float32)

        word_vec_list.append(word_vec)

    print ("Done!")
    return np.array(word_vec_list)