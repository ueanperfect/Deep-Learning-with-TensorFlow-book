import pandas as pd
import tensorflow as tf
import numpy as np

df = pd.read_excel('/Users/faguangnanhai/Desktop/excel_process.xlsx')

#data=df.loc[[0,1,2]].values
#print(data1)
#data=df.loc[[1,2],['title','data']].values
#print(data)

#data=df.loc[:,['title','data']].values#读所有行的title以及data列的值，这里需要嵌套列表
#print("读取指定行的数据：\n{0}".format(data))

#print("输出行号列表",df.index.values)

#print("输出列标题",df.columns.values)

#print("输出值",df.sample(3).values)#这个方法类似于head()方法以及df.values方法

#print("输出值\n",df['data'].values)

#pandas处理Excel数据成为字典

test_data=df.values
print(test_data)
test=tf.convert_to_tensor(test_data)
print(test)

# print(df.index)
# for i in df.index.values:#获取行号的索引，并对其进行遍历：
#     #根据i来获取每一行指定的数据 并利用to_dict转成字典
#     row_data=df.loc[i,['data']].to_dict()
#     test_data.append(row_data)
# print(test_data)