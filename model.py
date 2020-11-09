import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers import regression
from tflearn.data_utils import to_categorical
from read_data import rows

BATCH_SIZE = 32
IMG_SIZE = 28
N_CLASSES = 4
LR = 0.001
N_EPOCHS = 50

train_data = [] # dữ liệu training
train_label = [] # label của chúng

#Xay dung mo hinh cho 26 chu cai
for letter in rows:
    if (letter[0] == '0') or (letter[0] == '1') or (letter[0] == '2') or (letter[0] == '3') or (letter[0] == '4') or (letter[0] == '5') or (letter[0] == '6') or (letter[0] == '7') or (letter[0] == '8') or (letter[0] == '9') or (letter[0] == '10') or (letter[0] == '11') or (letter[0] == '12') or (letter[0] == '13') or (letter[0] == '14') or (letter[0] == '15') or (letter[0] == '16') or (letter[0] == '17') or (letter[0] == '18') or (letter[0] == '19') or (letter[0] == '20') or (letter[0] == '21') or (letter[0] == '22') or (letter[0] == '23') or (letter[0] == '24') or (letter[0] == '25') or (letter[0] == '26'):
        x = np.array([int(j) for j in letter[1:]])
        x = x.reshape(28, 28)
        train_data.append(x)
        train_label.append(int(letter[0]))
    else:
        break

#tong so anh thu duoc
#print(len(train_label))

#xao tron du lieu
import random
shuffle_order = list(range(56081))
random.shuffle(shuffle_order)
train_data = np.array(train_data)
train_label = np.array(train_label)

train_data = train_data[shuffle_order]
train_label = train_label[shuffle_order]


print(train_data.shape)
train_x = train_data[:50000]
train_y = train_label[:50000]

val_x = train_data[50000:53000]
val_y = train_label[50000:53000]

test_x = train_data[53000:]
test_y = train_label[53000:]

train_x = train_x.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
val_x = val_x.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_x = test_x.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

original_test_y = test_y
train_y = to_categorical(train_y, N_CLASSES)
val_y = to_categorical(val_y, N_CLASSES)
test_y = to_categorical(test_y, N_CLASSES)

tf.reset_default_graph()

network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1])  # 1

network = conv_2d(network, 32, 3, activation='relu')  # 2
network = max_pool_2d(network, 2)  # 3

network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)

network = fully_connected(network, 1024, activation='relu')  # 4
network = dropout(network, 0.8)  # 5

network = fully_connected(network, N_CLASSES, activation='softmax')  # 6
network = regression(network)

model = tflearn.DNN(network)  # 7

model.fit(train_x, train_y, n_epoch=N_EPOCHS, validation_set=(val_x, val_y), show_metric=True)

model.save('mymodel.tflearn')
