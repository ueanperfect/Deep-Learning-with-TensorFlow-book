from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics, preprocessing
import tensorflow as tf
from tensorflow import keras
import tensorboard
import numpy as np
import os
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
assert tf.__version__.startswith("2.")

# the most frequent word
batchsz = 10000
total_words = 10000
max_review_len = 80
embedding_len = 100
(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=total_words)
# x_train: (b, 80)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)

db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.shuffle(1000).batch(batchsz, drop_remainder=True)
print(x_train.dtype, x_test.shape, tf.reduce_max(x_train), tf.reduce_max(y_train))



class MyRNN(keras.Model):
    def __init__(self, units):
        super(MyRNN, self).__init__()
        # transform text to enbedding representation
        # [B, 80]->[B,80,100]
        self.state0 = [tf.zeros([batchsz, units])]
        self.embedding = layers.Embedding(total_words, embedding_len, input_length=max_review_len)
        # [b,80,100], h_dim:64
        # RNN: cell, cell2, cell3
        # simpleRNN
        self.rnn_cell0 = layers.SimpleRNNCell(units,)
        # self.rnn_cell1 = layers.SimpleRNN()
        # fc,[b,80,100]=>[b, 64]=>[b,1]
        self.rnn_fc = layers.Dense(1)

    def call(self, inputs, training=None):
        x = inputs
        # embedding [b,80]=>[b,80,q00]
        x = self.embedding(x)
        state0 = self.state0
        for word in tf.unstack(x, axis=1):  # word:[b,100]
            # h = tf.zeros(unit,)
            out, state1 = self.rnn_cell0(word, state0, training)
            state0 = state1
        x = self.rnn_fc(out)
        prob = tf.sigmoid(x)

        return prob

units = 64
epochs =5

model = MyRNN(units)
# model.build(input_shape=(4,80))
# model.summary()
model.compile(optimizer = keras.optimizers.Adam(0.001),
              loss = tf.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
model.fit(db_train, epochs=epochs, validation_data=db_test)
model.evaluate(db_test)
# log_dir="/Users/faguangnanhai/chb/jupyter/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)  # 定义TensorBoard对象
# model.fit(x=x_train,y=y_train,epochs=5,validation_data=None,callbacks=[tensorboard_callback])