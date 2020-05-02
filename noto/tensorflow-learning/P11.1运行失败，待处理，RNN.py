import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

total_words = 10000
max_review_len = 80
batchsz = 128
embedding_len = 100

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)

db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.shuffle(1000).batch(batchsz, drop_remainder=True)

print('x_train shape:', x_train.shape, tf.reduce_max(y_train), tf.reduce_min(y_train))
print('x test shape:', x_test.shape)


class MyRnn(keras.Model):

    def __init__(self, units):
        super(MyRnn, self).__init__()

        self.state0 = [tf.zeros([batchsz, units])]
        self.state1 = [tf.zeros([batchsz, units])]

        self.embedding = layers.Embedding(total_words, embedding_len,
                                          input_length=max_review_len)

        self.rnn_cell0 = layers.SimpleRNNCell(units, dropout=0.2)
        self.rnn_cell1 = layers.SimpleRNNCell(units, dropout=0.2)

        self.outlayer = layers.Dense(1)

    def call(self, inputs, training=None):
        x = self.embedding(inputs)

        state0 = self.state0
        state1 = self.state1
        for word in tf.unstack(x, axis=1):
            out0, state0 = self.rnn_cell0(word, state0, training)
            out1, state1 = self.rnn_cell1(out0, state1)

        x = self.outlayer(out1)

        prob = tf.sigmoid(x)

        return prob


def main():
    units = 64
    epoch = 4
    start_time = time.time()
    model = MyRnn(units)
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    model.fit(db_train, epochs=epoch, validation_data=db_test)
    model.evaluate(db_test)
    end_time = time.time()
    print('all time: ', end_time - start_time)


if __name__ == '__main__':
    main()


# 定义日志目录，必须是启动web应用时指定目录的子目录，建议使用日期时间作为子目录名

