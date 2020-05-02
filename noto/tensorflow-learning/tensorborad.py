import tensorflow as tf
import tensorboard
import datetime

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
  return tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

model = create_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 定义日志目录，必须是启动web应用时指定目录的子目录，建议使用日期时间作为子目录名
log_dir="/Users/faguangnanhai/chb/jupyter/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)  # 定义TensorBoard对象

model.fit(x=x_train,y=y_train,epochs=5,validation_data=(x_test, y_test),callbacks=[tensorboard_callback])  # 将定义好的TensorBoard对象作为回调传给fit方法，这样就将TensorBoard嵌入了模型训练过程