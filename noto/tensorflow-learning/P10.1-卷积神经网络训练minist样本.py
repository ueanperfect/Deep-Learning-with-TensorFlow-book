import tensorflow as tf
from tensorflow.keras import datasets, layers,Sequential,losses,optimizers

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def preprocess(x, y):
    # [b, 28, 28], [b]
    print(x.shape, y.shape)
    x = tf.cast(x, dtype=tf.float32)
    #x = tf.reshape(x, [-1, 28 * 28])
    y = tf.cast(y, dtype=tf.int32)
   # y = tf.one_hot(y, depth=10)
    return x, y

(x, y), (x_test, y_test) = datasets.mnist.load_data()
print('x:', x.shape, 'y:', y.shape, 'x test:', x_test.shape, 'y test:', y_test.shape)
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000)
train_db = train_db.batch(512).map(preprocess)
# %%
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.shuffle(1000).batch(512).map(preprocess)

x = tf.convert_to_tensor(x, dtype = tf.float32)
y = tf.convert_to_tensor(y, dtype=tf.int32)
x_test = tf.cast(x_test, dtype=tf.float32)
y_test = tf.cast(y_test, dtype=tf.int32)
x, y = next(iter(train_db))#将x，y设置成可迭代对象

network = Sequential([
    layers.Conv2D(6,kernel_size=3,strides=1),
    layers.MaxPooling2D(pool_size=2,strides=2),
    layers.ReLU(),
    layers.Conv2D(16,kernel_size=3,strides=1),
    layers.MaxPooling2D(pool_size=2,strides=2),
    layers.ReLU(),
    layers.Flatten(),
    layers.Dense(120,activation='relu'),
    layers.Dense(84,activation='relu'),
    layers.Dense(10)
])

network.build(input_shape=(4,28,28,1))
network.summary()

optimizer = tf.keras.optimizers.Adam(0.001)
criteon = losses.CategoricalCrossentropy(from_logits=True)#将softmax函数内置到softmax中

Epoch = 1
for step in range(Epoch):
    for step, (x, y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            x = tf.expand_dims(x,axis=3)
            out = network(x,training=True)
            y_onehot = tf.one_hot(y,depth=10)
            loss = criteon(y_onehot,out)
            grads = tape.gradient(loss, network.trainable_variables)
            optimizer.apply_gradients(zip(grads, network.trainable_variables))
    correct,total =0,0
    for x,y in test_db:
        x = tf.expand_dims(x,axis=3)
        out = network(x,training=False)
        pred = tf.argmax(out,axis=-1)
        y = tf.cast(y,tf.int64)
        correct += float(tf.reduce_sum(tf.cast(tf.equal(pred,y),tf.float32)))
        total += x.shape[0]
    print('test acc',correct/total)

print(network.variables)

