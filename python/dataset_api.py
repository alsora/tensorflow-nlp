import tensorflow as tf
import numpy as np


## Dataset tutorial https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428
## Iterator docs https://www.tensorflow.org/programmers_guide/datasets#creating_an_iterator
## NLP preprocessing pipeline example https://medium.com/@TalPerry/getting-text-into-tensorflow-with-the-dataset-api-ffb832c8bec6

#TENSORFLOW GRAPH

input_x = tf.placeholder(tf.string, [None], name="input_x")

dataset = tf.data.Dataset.from_tensor_slices(input_x)

iterator = dataset.make_initializable_iterator()

x = iterator.get_next()

tokens = tf.string_split([x])

keys = tf.constant(["this", "is", "a", "string","hello", "world", "not"])
values = tf.constant([0,1,2,3,4,5,6])

default_value = tf.constant(-1)

table = tf.contrib.lookup.HashTable(
    tf.contrib.lookup.KeyValueTensorInitializer(keys, values), default_value
)

out = table.lookup(tokens)



#MAIN PROGRAM
text = ["this is a UNK string", "hello world love this", "hello world", "this is not my string"]


session = tf.InteractiveSession()
table.init.run() #The table must be initialized within the session!


feed_dict = {
    input_x: text,
}

session.run(iterator.initializer, feed_dict={ input_x: text })

for _ in range(len(text)):
    output = session.run([out])
    print (output)