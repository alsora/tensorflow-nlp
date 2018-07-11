import tensorflow as tf
import numpy as np



#TENSORFLOW GRAPH

input_x = tf.placeholder(tf.string, [None], name="input_x")

# Create dataset from placeholder
dataset = tf.data.Dataset.from_tensor_slices(input_x)
# Shuffle the dataset at each epoch 
dataset  = dataset.shuffle(buffer_size=2)

# Divide dataset into batches of size BATCH_SIZE, padding along a dimension
BATCH_SIZE = 4
PAD_TOKEN = '_PADDING_'
padding_shapes = [10]
#dataset = dataset.padded_batch(BATCH_SIZE, padding_shapes, padding_values=PAD_TOKEN)
dataset = dataset.batch(BATCH_SIZE)
# Make the iterator restart after the dataset end
dataset = dataset.repeat()


def splitter_funct(x):
    splitted = tf.string_split(x)
    #padding_values = [[0,0], [0, 3]]
    #padded = tf.pad(splitted, padding_values, constant_values= PAD_TOKEN)
    return splitted

def padder_funct(x):
    print x
    padded = tf.concat([x, tf.constant["this"]], -1)
    return padded

dataset = dataset.map(splitter_funct)
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

out = table.lookup(x)



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