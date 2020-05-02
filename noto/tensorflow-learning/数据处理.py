import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf

from tensorflow import keras

input_filepath = '/Users/faguangnanhai/Desktop/shakespeare.txt'
text = open(input_filepath,'r').read()
#print(len(text))
#print(text[0:100])

vocab = sorted(set(text))
#print(vocab)
char2idx = {char:idx for idx,char in enumerate(vocab)}
#print(char2idx)#给每个字符赋值

idx2char = np.array(vocab)
#print(idx2char)

text_as_int = np.array([char2idx[c] for c in text])
#print(text_as_int)

def split_input_target(id_text):
    return id_text[0:-1],id_text[1:]

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
seq_length = 100
seq_dataset = char_dataset.batch(seq_length+1,drop_remainder=True)
for ch_id in char_dataset.take(1):
    print(ch_id,idx2char[ch_id.numpy()])

for seq_id in seq_dataset.take(1):
    print(seq_id)
    print(''.join(idx2char[seq_id.numpy()]))




