import math
import numpy as np

class Theta_grad():
    def __init__(self, Theta1_grad, Theta2_grad):
        self.Theta1_grad = Theta1_grad
        self.Theta2_grad = Theta2_grad


def sigmoid(z):
    g = 1/(1+math.e**(-z))
    g_prim = g*(1-g)
    return g_prim


def unpack_thetas(thetas, input_layer_size, hidden_layer_size, num_labels):
    t1_start = 0
    t1_end = hidden_layer_size * (input_layer_size + 1)
    t1 = thetas[t1_start:t1_end].reshape(
        (hidden_layer_size, input_layer_size + 1))
    t2 = thetas[t1_end:].reshape((num_labels, hidden_layer_size + 1))
    return t1, t2

def backpropagation(params, *args):
    X, Y, m, lmbd, input_layer_size, hidden_layer_size, num_labels = args

    Theta1, Theta2 = unpack_thetas(
        params, input_layer_size, hidden_layer_size, num_labels)
    #T_g = Theta_grad(np.zeros(Theta1.shape), np.zeros(Theta2.shape))   

    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    X = X.reshape([m, 784])
    X = np.concatenate((np.ones([m, 1]), X), axis=1)

    for t in range(1, m):
        a1 = np.transpose(X[t])
        z2 = np.dot(Theta1, a1)
        a2 = np.concatenate((np.ones([1]), sigmoid(z2)), axis=0)
        z3 = np.dot(Theta2, a2)
        a3 = sigmoid(z3)
        yVector = Y[t]
        delta3 = np.subtract(a3, yVector)
        delta2 = np.multiply((np.dot(np.transpose(Theta2), delta3)), np.concatenate(
            (np.ones([1]), sigmoid(z2)), axis=0))
        delta2 = np.delete(delta2, 1)
        Theta1_grad = np.add(
            Theta1_grad, (np.dot(delta2.reshape([delta2.shape[0], 1]), np.transpose(a1.reshape([a1.shape[0], 1])))))
        Theta2_grad = np.add(
            Theta2_grad, (np.dot(delta3.reshape([delta3.shape[0], 1]), np.transpose(a2.reshape([a2.shape[0], 1])))))

    Theta1_grad = (1/m)*Theta1_grad
    Theta2_grad = (1/m)*Theta2_grad

    for i in range(1, Theta1.shape[0]):
        for j in range(2, Theta1.shape[1]):
            Theta1_grad[i][j] = Theta1_grad[i][j] + \
                (lmbd/m)*Theta1[i][j]

    for i in range(1, Theta2.shape[0]):
        for j in range(2, Theta2.shape[1]):
            Theta2_grad[i][j] = Theta2_grad[i][j] + \
                (lmbd/m)*Theta2[i][j]

    T_g = np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten()))

    return T_g


def nnCostFunction(params, *args):
    X, Y, m, lmbd, input_layer_size, hidden_layer_size, num_labels = args

    Theta1, Theta2 = unpack_thetas(
        params, input_layer_size, hidden_layer_size, num_labels)

    #res = backpropagation(params, X, Y, m, lmbd, input_layer_size, hidden_layer_size, num_labels)

    X = X.reshape([m, 784])
    X = np.concatenate((np.ones([m, 1]), X), axis=1)

    newY = np.zeros([num_labels, m])
    for i in range(0, m):
        newY[Y[i]][i] = 1

    a1 = X
    z2 = np.dot(a1, np.transpose(Theta1))
    a2 = np.concatenate((np.ones([m, 1]), sigmoid(z2)), axis=1)
    z3 = np.dot(a2, np.transpose(Theta2))
    a3 = sigmoid(z3)
    hX = a3

    J = 0.0
    for i in range(1, m):
        for j in range(1, num_labels):
            J = J + (-newY[j][i]*np.log(hX[i][j]) -
                     (1-newY[j][i])*np.log(1-hX[i][j]))/m

    regT = 0.0
    for i in range(1, Theta1.shape[0]):
        for j in range(2, Theta1.shape[1]):
            regT = regT + Theta1[i][j]**2

    for i in range(1, Theta2.shape[0]):
        for j in range(2, Theta2.shape[1]):
            regT = regT + Theta2[i][j]**2

    regT = lmbd*regT/(2*m)

    J = J + regT

    print(J)

    return J
