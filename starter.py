import tensorflow as tf
import numpy as np


def loadData():
    with np.load('notMNIST.npz') as data:
        Data, Target = data['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target == posClass) + (Target == negClass)
        Data = Data[dataIndx] / 255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target == posClass] = 1
        Target[Target == negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget


def MSE(W, b, x, y, reg):
    loss_d = 0.5 * (((W * x).sum(axis=1) + b - y) ** 2).mean()
    loss_w = 0.5 * reg * (W ** 2).sum()
    return loss_d + loss_w


def gradMSE(W, b, x, y, reg):
    pass


def crossEntropyLoss(W, b, x, y, reg):
    pass


def gradCE(W, b, x, y, reg):
    pass


def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    pass


def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    pass
