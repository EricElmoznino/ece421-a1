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
    y_hat = (W * x).sum(axis=1) + b
    loss_d = 0.5 * ((y_hat - y) ** 2).mean()
    loss_w = 0.5 * reg * (W ** 2).sum()
    return loss_d + loss_w


def gradMSE(W, b, x, y, reg):
    pass


def crossEntropyLoss(W, b, x, y, reg):
    y_hat = (W * x).sum(axis=1) + b
    y_hat = 1 / (1 + np.exp(-y_hat))
    loss_d = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)).mean()
    loss_w = 0.5 * reg * (W ** 2).sum()
    return loss_d + loss_w


def gradCE(W, b, x, y, reg):
    pass


def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    pass


def buildGraph(beta1=0.9, beta2=0.999, epsilon=1e-8, lossType='MSE', learning_rate=0.001):
    tf.set_random_seed(421)

    x = tf.placeholder(tf.float32, shape=[None, 28 * 28])
    y = tf.placeholder(tf.float32, shape=[None])
    reg = tf.placeholder(tf.float32, shape=[])
    W = tf.Variable(tf.truncated_normal(shape=[28 * 28], stddev=0.5))
    b = tf.Variable([0.0])

    y_hat = tf.reduce_sum(W * x, axis=1) + b

    if lossType == "MSE":
        loss_d = 0.5 * tf.reduce_mean(((y_hat - y) ** 2))
    elif lossType == "CE":
        y_hat = 1 / (1 + tf.exp(-y_hat))
        loss_d = -tf.reduce_mean((y * tf.log(y_hat) + (1 - y) * tf.log(1 - y_hat)))
    else:
        raise ValueError('Unknown lossType:', lossType)
    loss_w = 0.5 * reg * tf.reduce_sum(W ** 2)
    loss = loss_d + loss_w

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
    optimize_op = optimizer.minimize(loss)

    return x, y_hat, y, W, b, loss, optimize_op, reg
