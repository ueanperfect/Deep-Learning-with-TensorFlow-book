#汽车油耗数据实战
##
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
##
dataset_path = keras.utils.get_file("auto-mpg.data","http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration','Model Year','Origin']
raw_dataset = pd.read_csv(dataset_path,names = column_names,na_values="?",comment='\t',sep=" ",skipinitialspace=True)
dataset = raw_dataset.copy()
#print(dataset.head())

##
#print(dataset.isna().sum())
dataset = dataset.dropna()#删除空白数据
#print(dataset.isna().sum())

origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1
dataset['Europe'] = (origin == 2)*1
dataset['Japan'] = (origin == 3)*1
dataset.tail()

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)
#print(dataset)
#print(train_dataset)
#print(test_dataset)

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

#sns.pairplot(train_dataset[["Cylinders", "Displacement", "Weight", "MPG"]],diag_kind="kde")

train_stats = train_dataset.describe()
#train_stats.pop('MPG')
#print(train_stats)
train_stats = train_stats.transpose()
#print(train_stats)
def norm(x):
    return(x-train_stats['mean'])/train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
print(normed_test_data)
##
#print(normed_train_data.values)
train_db = tf.data.Dataset.from_tensor_slices((normed_train_data.values,train_labels.values))
train_db = train_db.shuffle(100).batch(32)#32指的是一次运行32个数据
##
class Network(keras.Model):
    def __init__(self):
        super(Network,self).__init__()#李粤琰该打！
        self.fc1 = layers.Dense(6,activation='relu')
        self.fc2 = layers.Dense(3,activation='relu')
        self.fc3 = layers.Dense(1)

    def call(self,inputs,training=None,mask=None):

        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
##
model = Network()
model.build(input_shape=(None,9))#4是随意设计的batch量，9为输入特征值
model.summary()
optimizer = tf.keras.optimizers.RMSprop(0.001)

#train_mae_losses = []
#test_mae_losses = []
##
for epoch in range(200):
    for step, (x,y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            out = model(x)
            loss = tf.reduce_mean(losses.MSE(y,out))
            mae_loss = tf.reduce_mean(losses.MAE(y,out))

        if step % 10 == 0:
            print(epoch,step,float(loss),float(mae_loss))
        grads = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))

#     train_mae_losses.append(float(mae_loss))
#     out = model(tf.constant(normed_test_data.values))
#     test_mae_losses.append(tf.reduce_mean(losses.MAE(test_labels, out)))
#
# plt.figure()
# plt.xlabel('times')
# plt.ylabel('MAE')
# plt.plot(train_mae_losses, label='train')
#
# plt.show()
#
# plt.plot(test_mae_losses,label='test')
#
# plt.show()

# print(model)


