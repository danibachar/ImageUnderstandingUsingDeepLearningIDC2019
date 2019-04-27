import numpy as np
from .losses import *

class LogisticRegression():
    # Classifer that uses softmax + cross entropy loss

    def __init__(self, X, y):
        num_features = X.shape[1]
        num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
        self.W = 0.0001 * np.random.randn(num_features, num_classes)

    def predict(self, X):
        """
        Use the weight of the classifier to predict a label.
        Input: 2D array of size (num_instances, num_features).
        Output: 2D array of class predictions (num_instances, num_classes).
        """
        y_pred = np.zeros(X.shape[0])
        scores = X.dot(self.W)
        y_pred = np.argmax(scores, axis=1)
        return y_pred

    def calc_accuracy(self, X, y):
        accuracy = 0.0
        ###########################################################################
        #                          START OF YOUR CODE                             #
        ###########################################################################
        y_pred = self.predict(X)
        if len(y_pred) != len(y):
            print("Error with data dimenstion!")
            return accuracy

        accuracy = np.sum(y_pred == y) / len(y) * 100
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################

        return accuracy

    def train(self, X, y, learning_rate=1e-3, reg=0, num_iters=100, batch_size=200, verbose=False):

        num_instances, num_features = X.shape
        num_classesnum_classes = np.max(y) + 1

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            batch_ind = np.random.choice(num_instances, batch_size, replace=True)
            X_batch = X[batch_ind]
            y_batch = y[batch_ind]

            loss, grad = self.loss(X_batch, y_batch)
            loss_history.append(loss)
            self.W += - learning_rate * grad

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            if verbose and it % 100 == 0:
                print ('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def loss(self, X_batch, y_batch, reg=0):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
