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
        trainData, trainTarget = Data[:3500].reshape((-1, 28 * 28)), Target[:3500].reshape((-1))
        validData, validTarget = Data[3500:3600].reshape((-1, 28 * 28)), Target[3500:3600].reshape((-1))
        testData, testTarget = Data[3600:].reshape((-1, 28 * 28)), Target[3600:].reshape((-1))
    return trainData, validData, testData, trainTarget, validTarget, testTarget


def MSE(w, b, x, y, reg):
    y_hat = (w * x).sum(axis=1) + b
    loss_d = 0.5 * ((y_hat - y) ** 2).mean()
    loss_w = 0.5 * reg * (w ** 2).sum()
    return loss_d + loss_w


def gradMSE(w, b, x, y, reg):
    y_hat = (w * x).sum(axis=1) + b
    error = y_hat - y
    w_grad = (x * error.reshape((-1, 1)) + reg).mean(axis=0)
    b_grad = error.mean(axis=0)
    return w_grad, b_grad


def crossEntropyLoss(w, b, x, y, reg):
    y_hat = (w * x).sum(axis=1) + b
    y_hat = 1 / (1 + np.exp(-y_hat))
    loss_d = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)).mean()
    loss_w = 0.5 * reg * (w ** 2).sum()
    return loss_d + loss_w


def gradCE(w, b, x, y, reg):
    y_hat = (w * x).sum(axis=1) + b
    y_hat = (1 / (1 + np.e ** -y_hat))
    y, y_hat = y.reshape((-1, 1)), y_hat.reshape((-1, 1))
    w_grad = ((y_hat - y) * y_hat * (1 - y_hat) * x + reg).mean(axis=0)
    b_grad = ((y_hat - y) * y_hat * (1 - y_hat)).mean(axis=0)
    return w_grad, b_grad


def accuracy(w, b, x, y):
    y_hat = (w * x).sum(axis=1) + b
    y_hat_thresh = (y_hat > 0.5).astype(np.uint8)
    return (y_hat_thresh == y).sum() / y.shape[0]


def grad_descent(w, b, x, y, alpha, iterations, reg, epsilon, loss_type='MSE'):
    for _ in range(iterations):
        if loss_type == 'MSE':
            w_grad, b_grad = gradMSE(w, b, x, y, reg)
        elif loss_type == 'CE':
            w_grad, b_grad = gradCE(w, b, x, y, reg)
        else:
            raise ValueError('Unknown lossType:', loss_type)
        w, b = w - alpha * normalize(w_grad, epsilon), b - alpha * normalize(b_grad, epsilon)
    return w, b


def buildGraph(beta1=0.9, beta2=0.999, epsilon=1e-8, loss_type='MSE', learning_rate=0.001):
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


def normalize(x, epsilon=1e-7):
    return x / (np.linalg.norm(x) + epsilon)
