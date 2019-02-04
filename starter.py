import tensorflow as tf
import numpy as np


def load_data():
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
        trainData, trainTarget = Data[:3500].reshape((-1, 28 * 28)), Target[:3500].reshape((-1)).astype(np.float32)
        validData, validTarget = Data[3500:3600].reshape((-1, 28 * 28)), Target[3500:3600].reshape((-1)).astype(np.float32)
        testData, testTarget = Data[3600:].reshape((-1, 28 * 28)), Target[3600:].reshape((-1)).astype(np.float32)
    return trainData, validData, testData, trainTarget, validTarget, testTarget


def mse(w, b, x, y, reg):
    y_hat = (w * x).sum(axis=1) + b
    loss_d = 0.5 * ((y_hat - y) ** 2).mean()
    loss_w = 0.5 * reg * (w ** 2).sum()
    return loss_d + loss_w


def grad_mse(w, b, x, y, reg):
    y_hat = (w * x).sum(axis=1) + b
    error = y_hat - y
    w_grad = (x * error.reshape((-1, 1))).mean(axis=0) + reg
    b_grad = error.mean(axis=0)
    return w_grad, b_grad


def cross_entropy_loss(w, b, x, y, reg):
    y_hat = (w * x).sum(axis=1) + b
    y_hat = 1 / (1 + np.exp(-y_hat))
    loss_d = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)).mean()
    loss_w = 0.5 * reg * (w ** 2).sum()
    return loss_d + loss_w


def grad_ce(w, b, x, y, reg):
    y_hat = (w * x).sum(axis=1) + b
    y_hat = 1 / (1 + np.exp(-y_hat))
    error = y_hat - y
    w_grad = (x * error.reshape((-1, 1))).mean(axis=0) + reg
    b_grad = error.mean(axis=0)
    return w_grad, b_grad


def accuracy(w, b, x, y, ce=False):
    y_hat = (w * x).sum(axis=1) + b
    if ce:
        y_hat = 1 / (1 + np.exp(-y_hat))
    y_hat_thresh = (y_hat > 0.5).astype(np.uint8)
    return (y_hat_thresh == y).sum() / y.shape[0]


def grad_descent(w, b, x, y, alpha, iterations, reg, epsilon, loss_type='MSE'):
    for _ in range(iterations):
        if loss_type == 'MSE':
            w_grad, b_grad = grad_mse(w, b, x, y, reg)
        elif loss_type == 'CE':
            w_grad, b_grad = grad_ce(w, b, x, y, reg)
        else:
            raise ValueError('Unknown lossType:', loss_type)
        if np.linalg.norm(w_grad) > epsilon:
            w, b = w - alpha * w_grad, b - alpha * b_grad
    return w, b


def build_graph(beta1=0.9, beta2=0.999, epsilon=1e-8, loss_type='MSE', learning_rate=0.001):
    tf.set_random_seed(421)

    x = tf.placeholder(tf.float32, shape=[None, 28 * 28])
    y = tf.placeholder(tf.float32, shape=[None])
    reg = tf.placeholder(tf.float32, shape=[])
    w = tf.Variable(tf.truncated_normal(shape=[28 * 28], stddev=0.5))
    b = tf.Variable([0.0])

    y_hat = tf.reduce_sum(w * x, axis=1) + b

    if loss_type == "MSE":
        loss_d = 0.5 * tf.reduce_mean((y_hat - y) ** 2)
    elif loss_type == "CE":
        y_hat = tf.sigmoid(y_hat)
        loss_d = -tf.reduce_mean(y * tf.log(y_hat) + (1 - y) * tf.log(1 - y_hat))
    else:
        raise ValueError('Unknown lossType:', loss_type)
    loss_w = 0.5 * reg * tf.reduce_sum(w ** 2)
    loss = loss_d + loss_w

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
    optimize_op = optimizer.minimize(loss)

    return x, y_hat, y, w, b, loss, optimize_op, reg
