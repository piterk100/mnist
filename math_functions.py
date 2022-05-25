import math
import numpy as np


def sigmoid(z):
    g = 1/(1+math.e**(-z))
    g_prim = g*(1-g)
    return g_prim


def costFunction(X, Y, Theta1, Theta2, m, k, lmbd):
    X = X.reshape([m, 784])
    X = np.concatenate((np.ones([m, 1]), X), axis=1)

    newY = np.zeros([k, m])
    for i in range(0, m):
        newY[Y[i]][i] = 1

    a1 = X
    z2 = np.dot(a1, np.transpose(Theta1))
    a2 = np.concatenate((np.ones([m, 1]), sigmoid(z2)), axis=1)
    z3 = np.dot(a2, np.transpose(Theta2))
    a3 = sigmoid(z3)
    hX = a3

    J = 0
    for i in range(1, m):
        for j in range(1, k):
            J = J + (-newY[j][i]*np.log(hX[i][j]) -
                     (1-newY[j][i])*np.log(1-hX[i][j]))/m

    regT = 0
    for i in range(1, Theta1.shape[0]):
        for j in range(2, Theta1.shape[1]):
            regT = regT + Theta1[i][j]**2

    for i in range(1, Theta2.shape[0]):
        for j in range(2, Theta2.shape[1]):
            regT = regT + Theta2[i][j]**2

    regT = lmbd*regT/(2*m)

    J = J + regT

    return J


class Theta_grad():
    def __init__(self, Theta1_grad, Theta2_grad):
        self.Theta1_grad = Theta1_grad
        self.Theta2_grad = Theta2_grad


def backpropagation(X, Y, Theta1, Theta2, m, lmbd, num_labels):
    T_g = Theta_grad(np.zeros(Theta1.shape), np.zeros(Theta2.shape))

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
        T_g.Theta1_grad = np.add(
            T_g.Theta1_grad, (np.dot(delta2.reshape([delta2.shape[0], 1]), np.transpose(a1.reshape([a1.shape[0], 1])))))
        T_g.Theta2_grad = np.add(
            T_g.Theta2_grad, (np.dot(delta3.reshape([delta3.shape[0], 1]), np.transpose(a2.reshape([a2.shape[0], 1])))))

    T_g.Theta1_grad = (1/m)*T_g.Theta1_grad
    T_g.Theta2_grad = (1/m)*T_g.Theta2_grad

    for i in range(1, Theta1.shape[0]):
        for j in range(2, Theta1.shape[1]):
            T_g.Theta1_grad[i][j] = T_g.Theta1_grad[i][j] + \
                (lmbd/m)*Theta1[i][j]

    for i in range(1, Theta2.shape[0]):
        for j in range(2, Theta2.shape[1]):
            T_g.Theta2_grad[i][j] = T_g.Theta2_grad[i][j] + \
                (lmbd/m)*Theta2[i][j]

    return T_g
