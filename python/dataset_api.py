import tensorflow as tf
import numpy as np


def splitter_funct(x):
    print x
    splitted = tf.string_split(x)
    #padding_values = [[0,0], [0, 3]]
    #padded = tf.pad(splitted, padding_values, constant_values= PAD_TOKEN)
    print splitted
    return splitted

def padder_funct(x):
    print x
    padded = tf.concat([x, tf.constant("this")], -1)
    return padded

#TENSORFLOW GRAPH

input_x = tf.placeholder(tf.string, [None], name="input_x")

input_splitted = tf.string_split(input_x)
input_dense = tf.sparse_tensor_to_dense(input_splitted, default_value='_PADDING_')

# Create dataset from placeholder
dataset = tf.data.Dataset.from_tensor_slices(input_dense)
# Shuffle the dataset at each epoch 
dataset  = dataset.shuffle(buffer_size=10)


# Divide dataset into batches of size BATCH_SIZE, padding along a dimension
BATCH_SIZE = 2
PAD_TOKEN = '_PADDING_'


dataset = dataset.padded_batch(2, padded_shapes=[10], padding_values = PAD_TOKEN)

#dataset = dataset.batch(BATCH_SIZE)
# Make the iterator restart after the dataset end
dataset = dataset.repeat()

#dataset = dataset.map(splitter_funct)
#dataset = dataset.map(lambda tokens: tf.sparse_tensor_to_dense(tokens, default_value=PAD_TOKEN))


#paddings = tf.constant([[0, 0], [0, 2]])
#dataset = dataset.map(lambda tokens: tf.pad(tokens, paddings, "CONSTANT", constant_values = PAD_TOKEN))


#dataset = dataset.map(lambda tokens: tf.concat([tokens, [tf.constant("this")]], -1))

#dataset = dataset.map(splitter_funct)
#dataset = dataset.map(padder_funct)


iterator = dataset.make_initializable_iterator()

x = iterator.get_next()

#tokens = tf.string_split(x)

keys = tf.constant(["this", "is", "a", "string","hello", "world", "not"])
values = tf.constant([0,1,2,3,4,5,6])

default_value = tf.constant(-1)

table = tf.contrib.lookup.HashTable(
    tf.contrib.lookup.KeyValueTensorInitializer(keys, values), default_value
)

#out = table.lookup(x)
out = x


#MAIN PROGRAM
text = ["this is a UNK string", "hello world love this", "hello world", "this is not my string", "test", "string"]


session = tf.InteractiveSession()
table.init.run() #The table must be initialized within the session!


feed_dict = {
    input_x: text,
}

session.run(iterator.initializer, feed_dict={ input_x: text })

for _ in range(len(text)):
    output = session.run([out])
    print "--------------------------------"
    print "................................"
    print (output)