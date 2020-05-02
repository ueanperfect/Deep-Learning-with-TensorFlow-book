import tensorflow as tf

lr = 0.01

x = tf.random.normal()

w1 = tf.Variable(tf.random.truncated_normal([784,256],stddev=0.1))#正态分布
b1 = tf.Variable(tf.zeros([256]))

w2 = tf.Variable(tf.random.truncated_normal([256,128],stddev=0.1))#正态分布
b2 = tf.Variable(tf.zeros([128]))

w3 = tf.Variable(tf.random.truncated_normal([256,10],stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

x = tf.reshape(x,[-1.28*28])

with tf.GradientTape() as tape:

    h1 = x@w1+tf.broadcast_to(b1,[x.shape[0],256])
    h1 = tf.nn.relu(h1)

    h2 = h1@w2+tf.broadcast_to(b2,[h1.shape[0],128])
    h2 = tf.nn.relu(h2)

    h3 = tf.broadcast_to(b3,[h2.shape[0],256])

    loss = tf.square(y_onehot -out)
    loss = tf.reduce_mean(loss)

    grads = tape.gradient(loss,[w1,b1,w2,b2,w3,b3])
    w1.assign_sub(lr*grads[0])
    b1.assign_sub(lr*grads[1])
    w2.assign_sub(lr*grads[2])
    b2.assign_sub(lr*grads[3])
    w3.assign_sub(lr*grads[4])
    b3.assign_sub(lr*grads[5])

