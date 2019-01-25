import unittest
from tqdm import tqdm
import os

from starter import *
from plotting import *


class Part1(unittest.TestCase):

    def setUp(self):
        self.train_x, self.val_x, self.test_x, self.train_y, self.val_y, self.test_y = loadData()

    def test_3(self):
        alphas = [0.005, 0.001, 0.0001]
        metrics = [{'Training loss': [], 'Validation loss': [], 'Test loss': [],
                    'Training accuracy': [], 'Validation accuracy': [], 'Test accuracy': []}
                   for _ in alphas]
        epochs = 5000
        save_freq = 100
        iterations = epochs // save_freq
        for alpha, metric in zip(alphas, metrics):
            print('Training with alpha=%g' % alpha)
            w, b = initialize()
            for _ in tqdm(range(0, epochs + 1, save_freq)):
                metric['Training loss'].append(MSE(w, b, self.train_x, self.train_y, 0.0))
                metric['Validation loss'].append(MSE(w, b, self.val_x, self.val_y, 0.0))
                metric['Test loss'].append(MSE(w, b, self.test_x, self.test_y, 0.0))
                metric['Training accuracy'].append(accuracy(w, b, self.train_x, self.train_y))
                metric['Validation accuracy'].append(accuracy(w, b, self.val_x, self.val_y))
                metric['Test accuracy'].append(accuracy(w, b, self.test_x, self.test_y))
                w, b = grad_descent(w, b, self.train_x, self.train_y, alpha, iterations, 0.0, 1e-7, 'MSE')
        for title in metrics[0]:
            line_plot(title, list(range(0, epochs + 1, save_freq)),
                      [m[title] for m in metrics], ['learning rate = %g' % a for a in alphas],
                      'epochs', title.split(' ')[1],
                      os.path.join('results', '1_3', title + '.png'))
        for a, m in zip(alphas, metrics):
            with open(os.path.join('results', '1_3', 'final_metrics_alpha=%g.txt' % a), 'w') as f:
                for title in m:
                    f.write('%s: %g\n' % (title, m[title][-1]))

    def test_4(self):
        regs = [0.001, 0.1, 0.5]
        metrics = [{'Training loss': [], 'Validation loss': [], 'Test loss': [],
                    'Training accuracy': [], 'Validation accuracy': [], 'Test accuracy': []}
                   for _ in regs]
        epochs = 5000
        save_freq = 100
        iterations = epochs // save_freq
        for reg, metric in zip(regs, metrics):
            print('Training with reg=%g' % reg)
            w, b = initialize()
            for _ in tqdm(range(0, epochs + 1, save_freq)):
                metric['Training loss'].append(MSE(w, b, self.train_x, self.train_y, 0.0))
                metric['Validation loss'].append(MSE(w, b, self.val_x, self.val_y, 0.0))
                metric['Test loss'].append(MSE(w, b, self.test_x, self.test_y, 0.0))
                metric['Training accuracy'].append(accuracy(w, b, self.train_x, self.train_y))
                metric['Validation accuracy'].append(accuracy(w, b, self.val_x, self.val_y))
                metric['Test accuracy'].append(accuracy(w, b, self.test_x, self.test_y))
                w, b = grad_descent(w, b, self.train_x, self.train_y, 0.005, iterations, reg, 1e-7, 'MSE')
        for title in metrics[0]:
            line_plot(title, list(range(0, epochs + 1, save_freq)),
                      [m[title] for m in metrics], ['regularization = %g' % a for a in regs],
                      'epochs', title.split(' ')[1],
                      os.path.join('results', '1_4', title + '.png'))
        for a, m in zip(regs, metrics):
            with open(os.path.join('results', '1_4', 'final_metrics_reg=%g.txt' % a), 'w') as f:
                for title in m:
                    f.write('%s: %g\n' % (title, m[title][-1]))

    def test_5(self):
        def closed_form_optimal(w, b, x, y):
            return w, b
        w, b = initialize()
        w, b = closed_form_optimal(w, b, self.train_x, self.train_y)
        metrics = {'Training loss': MSE(w, b, self.train_x, self.train_y, 0.0),
                   'Validation loss': MSE(w, b, self.val_x, self.val_y, 0.0),
                   'Test loss': MSE(w, b, self.test_x, self.test_y, 0.0),
                   'Training accuracy': accuracy(w, b, self.train_x, self.train_y),
                   'Validation accuracy': accuracy(w, b, self.val_x, self.val_y),
                   'Test accuracy': accuracy(w, b, self.test_x, self.test_y)}
        with open(os.path.join('results', '1_5', 'final_metrics.txt'), 'w') as f:
            for title in metrics:
                f.write('%s: %g\n' % (title, metrics[title]))


def initialize():
    w = np.zeros(28 * 28)
    b = np.zeros(1)
    return w, b
