# encoding: utf-8

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, losses, optimizers, datasets
import matplotlib.pyplot as plt

Epoch = 10
path = r'G:\2019\python\mnist.npz'
(x, y), (x_val, y_val) = datasets.mnist.load_data()#tf.keras.datasets.mnist.load_data(path)  # 60000 and 10000
print('datasets:', x.shape, y.shape, x.min(), x.max())

x = tf.convert_to_tensor(x, dtype = tf.float32)  #/255.    #0:1  ;   -1:1(不适合训练，准确度不高)
# x = tf.reshape(x, [-1, 28*28])
y = tf.convert_to_tensor(y, dtype=tf.int32)
# y = tf.one_hot(y, depth=10)
#将60000组训练数据切分为600组，每组100个数据
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(60000)      #尽量与样本空间一样大
train_db = train_db.batch(100)          #128

x_val = tf.cast(x_val, dtype=tf.float32)
y_val = tf.cast(y_val, dtype=tf.int32)
test_db = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_db = test_db.shuffle(10000)
test_db = test_db.batch(100)        #128

network = Sequential([
    layers.Conv2D(6, kernel_size=3, strides=1),  # 6个卷积核
    layers.MaxPooling2D(pool_size=2, strides=2),  # 池化层，高宽各减半
    layers.ReLU(),
    layers.Conv2D(16, kernel_size=3, strides=1),  # 16个卷积核
    layers.MaxPooling2D(pool_size=2, strides=2),  # 池化层，高宽各减半
    layers.ReLU(),
    layers.Flatten(),
    #layers.Dense(120, activation='relu'),
    #layers.Dense(84, activation='relu'),
    layers.Dense(10)
])
network.build(input_shape=(4, 28, 28, 1))
network.summary()
optimizer = tf.keras.optimizers.Adam(0.001)              # 创建优化器，指定学习率
criteon = losses.CategoricalCrossentropy(from_logits=True)

# 保存训练和测试过程中的误差情况
train_tot_loss = []
test_tot_loss = []

for step in range(Epoch):
    cor, tot = 0, 0
    for x, y in train_db:
        with tf.GradientTape() as tape:  # 构建梯度环境
            # 插入通道维度 [None,28,28] -> [None,28,28,1]
            x = tf.expand_dims(x, axis=3)
            out = network(x)
            y_true = tf.one_hot(y, 10)
            loss =criteon(y_true, out)

            out_train = tf.argmax(out, axis=-1)
            y_train = tf.cast(y, tf.int64)
            cor += float(tf.reduce_sum(tf.cast(tf.equal(y_train, out_train), dtype=tf.float32)))
            tot += x.shape[0]

            grads = tape.gradient(loss, network.trainable_variables)
            optimizer.apply_gradients(zip(grads, network.trainable_variables))
    print('After %d Epoch' % step)
    print('training acc is ', cor/tot)
    train_tot_loss.append(cor/tot)

    correct, total = 0, 0
    for x, y in test_db:
        x = tf.expand_dims(x, axis=3)
        out = network(x)
        pred = tf.argmax(out, axis=-1)
        y = tf.cast(y, tf.int64)
        correct += float(tf.reduce_sum(tf.cast(tf.equal(y, pred), dtype=tf.float32)))
        total += x.shape[0]
    print('testing acc is : ', correct/total)
    test_tot_loss.append(correct/total)


plt.figure()
plt.plot(train_tot_loss, 'b', label='train')
plt.plot(test_tot_loss, 'r', label='test')
plt.xlabel('Epoch')
plt.ylabel('ACC')
plt.legend()
plt.savefig('exam8.2_train_test_CNN1.png')
plt.show()