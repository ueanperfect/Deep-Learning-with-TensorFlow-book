import tensorflow as tf
x = tf.random.normal([4,28*28])
from tensorflow.keras import layers # 导入层模块
# 创建全连接层，指定输出节点数和激活函数
fc = layers.Dense(512, activation=tf.nn.relu)
h1 = fc(x) # 通过 fc 类实例完成一次全连接层的计算，返回输出张量

h2=fc.trainable_variables


print(h2)