# Loading data

import math_functions as mf
import numpy as np
from scipy import optimize

f_train_images = open('train-images.idx3-ubyte', 'rb')
f_train_labels = open('train-labels.idx1-ubyte', 'rb')

image_size = 28
train_set_size = 10000

f_train_images.read(16)
buf = f_train_images.read(image_size * image_size * train_set_size)
train_images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
train_images = train_images.reshape(train_set_size, image_size, image_size)

f_train_labels.read(16)
buf = f_train_labels.read(train_set_size)
train_labels = np.frombuffer(buf, dtype=np.uint8).astype(np.uint8)
train_labels = train_labels.reshape(train_set_size)

#import matplotlib.pyplot as plt
#image = np.asarray(train_images[1000]).squeeze()
# plt.imshow(image)
# plt.show()

#print(train_images[0, 14, 14])

# Neural network

input_layer_size = image_size * image_size
hidden_layer_size = 25
num_labels = 10


Theta1 = np.random.rand(25, 785) % 0.24 - 0.12
Theta2 = np.random.rand(10, 26) % 0.24 - 0.12

# print(mf.nnCostFunction(train_images, train_labels,
#      Theta1, Theta2, train_set_size, num_labels, 1))

# mf.backpropagation(train_images, train_labels, Theta1,
#                   Theta2, train_set_size, 1)

initial_nn_params = np.concatenate((Theta1.flatten(), Theta2.flatten()))

lmbda = 1

args = (train_images, train_labels, train_set_size, lmbda,
        input_layer_size, hidden_layer_size, num_labels)


def costFunction(p, *args):
    return mf.nnCostFunction(p, *args)


def backProp(p, *args):
    return mf.backpropagation(p, *args)


res1 = optimize.fmin_cg(costFunction, initial_nn_params,
                        fprime=backProp, args=args)

print(res1)
